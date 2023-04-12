import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from tensorflow import keras

from keras_aug.utils import augmentation_utils

H_AXIS = -3
W_AXIS = -2


@keras.utils.register_keras_serializable(package="keras_aug")
class PadIfNeeded(VectorizedBaseImageAugmentationLayer):
    """Pad the images with zeros to ensure that all images within the same
    batch are of the same size.

    This layer can be configured by specifying the fixed sizes or the divisors.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.

    Args:
        min_height: A integer specifying the height of result image.
        min_width: A integer specifying the width of result image.
        height_divisor: A integer that ensures image height is dividable
            by this value.
        width_divisor: A integer that ensures image width is dividable
            by this value.
        position: A string specifying the padding method, defaults
            to "center".
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        min_height=None,
        min_width=None,
        pad_height_divisor=None,
        pad_width_divisor=None,
        position="center",
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if (min_height is None or min_width is None) and (
            pad_height_divisor is None or pad_width_divisor is None
        ):
            raise ValueError(
                "`PadIfNeeded()` expects at most one of "
                "(`min_height`, `min_width`) or "
                "(`pad_height_divisor`, `pad_width_divisor`) to be set."
            )

        self.min_height = min_height
        self.min_width = min_width
        self.pad_height_divisor = pad_height_divisor
        self.pad_width_divisor = pad_width_divisor
        self.position = augmentation_utils.get_padding_position(position)
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(images)

        if self.min_height is not None:
            h_tops = tf.where(
                heights < self.min_height,
                tf.cast((self.min_height - heights) / 2, heights.dtype),
                tf.zeros_like(heights, dtype=heights.dtype),
            )
            h_bottoms = tf.where(
                heights < self.min_height,
                self.min_height - heights - h_tops,
                tf.zeros_like(heights, dtype=heights.dtype),
            )
        else:
            pad_remaineds = heights % self.pad_height_divisor
            pad_rows = tf.where(
                tf.math.greater(pad_remaineds, 0),
                self.pad_height_divisor - pad_remaineds,
                tf.zeros_like(pad_remaineds, dtype=pad_remaineds.dtype),
            )
            pad_rows = tf.cast(pad_rows, dtype=tf.float32)
            h_tops = tf.round(pad_rows / 2.0)
            h_bottoms = tf.cast(pad_rows - h_tops, dtype=tf.int32)
            h_tops = tf.cast(h_tops, dtype=tf.int32)

        if self.min_width is not None:
            w_lefts = tf.where(
                widths < self.min_width,
                tf.cast((self.min_width - widths) / 2, widths.dtype),
                tf.zeros_like(widths, dtype=widths.dtype),
            )
            w_rights = tf.where(
                widths < self.min_width,
                self.min_width - widths - w_lefts,
                tf.zeros_like(widths, dtype=widths.dtype),
            )
        else:
            pad_remaineds = widths % self.pad_width_divisor
            pad_cols = tf.where(
                tf.math.greater(pad_remaineds, 0),
                self.pad_width_divisor - pad_remaineds,
                tf.zeros_like(pad_remaineds, dtype=pad_remaineds.dtype),
            )
            pad_cols = tf.cast(pad_cols, dtype=tf.float32)
            w_lefts = tf.round(pad_cols / 2.0)
            w_rights = tf.cast(pad_cols - w_lefts, dtype=tf.int32)
            w_lefts = tf.cast(w_lefts, dtype=tf.int32)

        h_tops, h_bottoms, w_lefts, w_rights = self._get_position_params(
            h_tops, h_bottoms, w_lefts, w_rights, self.position
        )

        return {
            "pad_tops": h_tops,
            "pad_bottoms": h_bottoms,
            "pad_lefts": w_lefts,
            "pad_rights": w_rights,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        pad_top = transformations["pad_tops"][0][0]
        pad_bottom = transformations["pad_bottoms"][0][0]
        pad_left = transformations["pad_lefts"][0][0]
        pad_right = transformations["pad_rights"][0][0]
        paddings = tf.stack(
            (
                tf.zeros(shape=(2,), dtype=pad_top.dtype),
                tf.stack((pad_top, pad_bottom)),
                tf.stack((pad_left, pad_right)),
                tf.zeros(shape=(2,), dtype=pad_top.dtype),
            )
        )
        images = tf.pad(images, paddings=paddings)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformations,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`PadIfNeeded()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`PadIfNeeded(..., bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )

        pad_tops = tf.cast(transformations["pad_tops"], dtype=tf.float32)
        pad_lefts = tf.cast(transformations["pad_lefts"], dtype=tf.float32)
        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        x1s += tf.expand_dims(pad_lefts, axis=1)
        y1s += tf.expand_dims(pad_tops, axis=1)
        x2s += tf.expand_dims(pad_lefts, axis=1)
        y2s += tf.expand_dims(pad_tops, axis=1)
        outputs = tf.concat([x1s, y1s, x2s, y2s], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=images,
        )
        return bounding_boxes

    def _get_position_params(
        self, h_tops, h_bottoms, w_lefts, w_rights, position
    ):
        """This function supposes arguments are at `center` padding method."""
        if position == augmentation_utils.PaddingPosition.CENTER:
            # do nothing
            h_bottoms = h_bottoms
            w_rights = w_rights
            h_tops = h_tops
            w_lefts = w_lefts
        elif position == augmentation_utils.PaddingPosition.TOP_LEFT:
            h_bottoms += h_tops
            w_rights += w_lefts
            h_tops = tf.zeros_like(h_tops)
            w_lefts = tf.zeros_like(w_lefts)
        elif position == augmentation_utils.PaddingPosition.TOP_RIGHT:
            h_bottoms += h_tops
            w_lefts += w_rights
            h_tops = tf.zeros_like(h_tops)
            w_rights = tf.zeros_like(w_rights)
        elif position == augmentation_utils.PaddingPosition.BOTTOM_LEFT:
            h_tops += h_bottoms
            w_rights += w_lefts
            h_bottoms = tf.zeros_like(h_bottoms)
            w_lefts = tf.zeros_like(w_lefts)
        elif position == augmentation_utils.PaddingPosition.BOTTOM_RIGHT:
            h_tops += h_bottoms
            w_lefts += w_rights
            h_bottoms = tf.zeros_like(h_bottoms)
            w_rights = tf.zeros_like(w_rights)
        elif position == augmentation_utils.PaddingPosition.RANDOM:
            batch_size = tf.shape(h_tops)[0]
            original_dtype = h_tops.dtype
            h_pads = tf.cast(h_tops + h_bottoms, dtype=tf.float32)
            w_pads = tf.cast(w_lefts + w_rights, dtype=tf.float32)
            h_tops = self._random_generator.random_uniform(
                shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
            )
            h_tops = tf.cast(tf.round(h_tops * h_pads), dtype=original_dtype)
            h_bottoms = tf.cast(h_pads, dtype=tf.int32) - h_tops
            w_lefts = self._random_generator.random_uniform(
                shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
            )
            w_lefts = tf.cast(tf.round(w_lefts * w_pads), dtype=original_dtype)
            w_rights = tf.cast(w_pads, dtype=tf.int32) - w_lefts
        else:
            raise NotImplementedError(
                f"Value not recognized for `position`: {position}. Supported "
                f"values are: {augmentation_utils.PADDING_POSITION}"
            )

        return h_tops, h_bottoms, w_lefts, w_rights

    def get_config(self):
        config = {
            "min_height": self.min_height,
            "min_width": self.min_width,
            "pad_height_divisor": self.pad_height_divisor,
            "pad_width_divisor": self.pad_width_divisor,
            "position": self.position,
            "bounding_box_format": self.bounding_box_format,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
