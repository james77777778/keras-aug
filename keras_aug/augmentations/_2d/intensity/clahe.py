import tensorflow as tf
from tensorflow import keras

from keras_aug.augmentations._2d.keras_aug_2d_base_layer import (
    KerasAug2DBaseLayer,
)
from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class CLAHE(KerasAug2DBaseLayer):
    """Apply Contrast Limited Adaptive Histogram Equalization to the input
    image.

    This layer expects input images with int32 types.

    Args:
        clip_limit: upper threshold value for contrast limiting.
            If clip_limit is a single float value, the range will be
            (1, clip_limit), defaults to (1, 4).
        tile_grid_size: A tuple of int representing the size of grid for
            histogram equalization, defaults to (8, 8).
        gpu_optimized: A bool specifying whether or not to use functions that
            perform better when XLA-compiled on the GPU, but worse on the CPU.
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        clip_limit=(1, 4),
        tile_grid_size=(8, 8),
        gpu_optimized=True,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.clip_limit = augmentation_utils.to_tuple(clip_limit, low=1)
        self.tile_grid_size = tuple(tile_grid_size)
        self.gpu_optimized = gpu_optimized

    def get_random_transformation_batch(self, batch_size, **kwargs):
        clip_limits = self._random_generator.random_uniform(
            shape=(batch_size, 1),
            minval=self.clip_limit[0],
            maxval=self.clip_limit[1],
            dtype=tf.float32,
        )
        return clip_limits

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        inputs_for_clahe_single_image = {
            "images": images,
            "clip_limits": transformations,
        }
        clahed_images = tf.map_fn(
            self.clahe_single_image,
            inputs_for_clahe_single_image,
            fn_output_signature=images.dtype,
        )
        return tf.cast(clahed_images, self.compute_dtype)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(self, bounding_boxes, transformations, **kwargs):
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        return segmentation_masks

    def augment_keypoints(self, keypoints, transformations, **kwargs):
        return keypoints

    def clahe_single_image(self, inputs):
        """Hardly borrow from:
        https://github.com/isears/tf_clahe
        """
        image = inputs.get("images", None)
        clip_limit = inputs.get("clip_limits", None)

        original_2d_shape = (tf.shape(image)[0], tf.shape(image)[1])
        original_dtype = image.dtype

        # Need image in int32 format for later gather_nd ops
        image = tf.cast(image, tf.int32)

        tile_shape = tf.truediv(original_2d_shape, self.tile_grid_size)
        tile_shape = tf.cast(tf.math.ceil(tile_shape), tf.int32)

        # Reflection-pad image
        pad_y = 0
        pad_x = 0
        if original_2d_shape[0] % tile_shape[0] != 0:
            pad_y = tile_shape[0] - (original_2d_shape[0] % tile_shape[0])
        if original_2d_shape[1] % tile_shape[1] != 0:
            pad_x = tile_shape[1] - (original_2d_shape[1] % tile_shape[1])

        image_padded = tf.pad(
            image, [[0, pad_y], [0, pad_x], [0, 0]], "REFLECT"
        )
        all_tiles = tf.space_to_batch(
            input=tf.expand_dims(image_padded, axis=0),
            block_shape=tile_shape,
            paddings=[[0, 0], [0, 0]],
        )

        # Compute per-tile histogram
        if self.gpu_optimized:
            hists = tf.math.reduce_sum(
                tf.one_hot(
                    all_tiles, depth=256, on_value=1, off_value=0, axis=0
                ),
                axis=1,
            )
        else:
            single_dimension_tiles = tf.reshape(
                all_tiles,
                (
                    tile_shape[0] * tile_shape[1],
                    self.tile_grid_size[0]
                    * self.tile_grid_size[1]
                    * tf.shape(image)[-1],
                ),
            )

            single_dimension_tiles = tf.transpose(single_dimension_tiles)
            hists = tf.math.bincount(
                single_dimension_tiles, minlength=256, maxlength=256, axis=-1
            )

            hists = tf.transpose(hists)
            hists = tf.reshape(
                hists,
                (
                    256,
                    self.tile_grid_size[0],
                    self.tile_grid_size[1],
                    tf.shape(image)[-1],
                ),
            )

        if clip_limit > 0:
            clip_limit_actual = clip_limit * tf.cast(
                ((tile_shape[0] * tile_shape[1]) / 256), tf.float32
            )
            clip_limit_actual = tf.cast(clip_limit_actual, tf.int32)
            clipped_hists = tf.clip_by_value(
                hists, clip_value_min=0, clip_value_max=clip_limit_actual
            )
            clipped_px_count = tf.math.reduce_sum(hists - clipped_hists, axis=0)
            clipped_hists = tf.cast(clipped_hists, tf.float32)
            clipped_px_count = tf.cast(clipped_px_count, tf.float32)
            clipped_hists = clipped_hists + tf.math.truediv(
                clipped_px_count, 256
            )
        else:
            clipped_hists = tf.cast(hists, tf.float32)

        cdf = tf.math.cumsum(clipped_hists, axis=0)
        cdf_min = tf.math.reduce_min(cdf, axis=0)
        numerator = cdf - cdf_min
        denominator = (
            tf.cast(tile_shape[0] * tile_shape[1], tf.float32) - cdf_min
        )
        cdf_normalized = tf.round(
            tf.math.divide_no_nan(numerator, denominator) * (255)
        )
        cdf_normalized = tf.cast(cdf_normalized, tf.int32)

        # Reflection-pad the cdf functions so that we don't have to explicitly
        # deal with corners/edges
        cdf_padded = tf.pad(
            cdf_normalized, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC"
        )
        coords = tf.stack(
            tf.meshgrid(
                tf.range(tf.shape(image_padded)[0]),
                tf.range(tf.shape(image_padded)[1]),
                tf.range(tf.shape(image_padded)[2]),
                indexing="ij",
            )
        )

        y_coords = coords[0, :, :]
        x_coords = coords[1, :, :]
        z_coords = coords[2, :, :]

        half_tile_shape = tf.math.floordiv(tile_shape, 2)

        nw_y_component = tf.math.floordiv(
            y_coords - half_tile_shape[0], tile_shape[0]
        )
        nw_x_component = tf.math.floordiv(
            x_coords - half_tile_shape[1], tile_shape[1]
        )

        # Need to correct negative values because negative-indexing for
        # gather_nd ops not supported on all processors
        # (cdf is padded to account for this)
        nw_y_component = nw_y_component + 1
        nw_x_component = nw_x_component + 1
        ne_y_component = nw_y_component
        ne_x_component = nw_x_component + 1
        sw_y_component = nw_y_component + 1
        sw_x_component = nw_x_component
        se_y_component = sw_y_component
        se_x_component = sw_x_component + 1

        def cdf_transform(x_comp, y_comp):
            gatherable = tf.stack(
                [image_padded, y_comp, x_comp, z_coords], axis=-1
            )
            return tf.cast(tf.gather_nd(cdf_padded, gatherable), tf.float32)

        nw_transformed = cdf_transform(nw_x_component, nw_y_component)
        ne_transformed = cdf_transform(ne_x_component, ne_y_component)
        sw_transformed = cdf_transform(sw_x_component, sw_y_component)
        se_transformed = cdf_transform(se_x_component, se_y_component)

        a = (y_coords - half_tile_shape[0]) % tile_shape[0]
        a = tf.cast(tf.math.truediv(a, tile_shape[0]), tf.float32)
        b = (x_coords - half_tile_shape[1]) % tile_shape[1]
        b = tf.cast(tf.math.truediv(b, tile_shape[1]), tf.float32)

        # Interpolate
        interpolated = (a * (b * se_transformed + (1 - b) * sw_transformed)) + (
            1 - a
        ) * (b * ne_transformed + (1 - b) * nw_transformed)

        # Return image to original size and dtype
        interpolated = interpolated[
            0 : original_2d_shape[0], 0 : original_2d_shape[1], :
        ]
        interpolated = tf.cast(tf.round(interpolated), original_dtype)
        return interpolated

    def get_config(self):
        config = super().get_config()
        config.update({"blur_limit": self.blur_limit, "seed": self.seed})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
