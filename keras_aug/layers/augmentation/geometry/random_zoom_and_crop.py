import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils import bounding_box as bounding_box_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomZoomAndCrop(VectorizedBaseRandomLayer):
    """RandomZoomAndCrop implements resize with scale distortion.

    RandomZoomAndCrop takes a three-step approach to size-distortion based image
    augmentation. This technique is specifically tuned for object detection
    pipelines. The layer takes an input of images and bounding boxes, both of
    which may be ragged. It outputs a dense image tensor, ready to feed to a
    model for training. As such this layer will commonly be the final step in an
    augmentation pipeline.

    The augmentation process is as follows:
    The image is first scaled according to a randomly sampled scale factor. The
    width and height of the image are then resized according to the sampled
    scale. This is done to introduce noise into the local scale of features in
    the image. A subset of the image is then cropped randomly according to
    ``(crop_height, crop_width)``. This crop is then padded to be
    ``(height, width)``. Bounding boxes are translated and scaled according to
    the random scaling and random cropping.

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        scale_factor (float|Sequence[float]|keras_aug.FactorSampler): The range
            of the scale factor that is used to scale the input image. When
            represented as a single float, the factor will be picked between
            ``[1.0 - lower, 1.0 + upper]``. To reproduce the results of the
            MaskRCNN paper pass ``(0.8, 1.25)``.
        crop_height (int, optional): The height of the image to crop from the
            scaled image. Defaults to ``height`` when not provided.
        crop_width (int, optional): The width of the image to crop from the
            scaled image. Defaults to ``width`` when not provided.
        interpolation (str, optional): The interpolation method. Supported values:
            ``"nearest", "bilinear", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"``.
            Defaults to ``"bilinear"``.
        antialias (bool, optional): Whether to use antialias. Defaults to
            ``False``.
        position (str, optional): The padding method.
            Supported values: ``"center", "top_left", "top_right", "bottom_left", "bottom_right", "random"``.
            Defaults to ``"center"``.
        padding_value (int|float, optional): The padding value.
            Defaults to ``0``.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        height,
        width,
        scale_factor,
        crop_height=None,
        crop_width=None,
        interpolation="bilinear",
        antialias=False,
        position="center",
        padding_value=0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                "RandomZoomAndCrop() expects ``height`` and ``width`` to be "
                f"integers. Received ``height={height}, width={width}``"
            )
        self.height = height
        self.width = width
        self.crop_height = crop_height or height
        self.crop_width = crop_width or width
        self.scale_factor = augmentation_utils.parse_factor(
            scale_factor,
            min_value=0.0,
            max_value=None,
            center_value=1.0,
            seed=seed,
        )
        self.interpolation = preprocessing_utils.get_interpolation(
            interpolation
        )
        self.antialias = antialias
        self.position = augmentation_utils.get_padding_position(position)
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        crop_size = tf.expand_dims(
            tf.stack([self.crop_height, self.crop_width]), axis=0
        )
        self.crop_size = tf.cast(crop_size, dtype=tf.float32)
        self.force_output_dense_images = True

    def compute_ragged_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=(self.height, self.width, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # cast to float32 to avoid numerical issue
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )

        # image_scales
        image_shapes = tf.concat((heights, widths), axis=-1)
        scaled_sizes = tf.round(
            image_shapes
            * self.scale_factor(shape=(batch_size, 1), dtype=tf.float32)
        )
        scales = tf.where(
            tf.less(
                scaled_sizes[..., 0:1] / image_shapes[..., 0:1],
                scaled_sizes[..., 1:] / image_shapes[..., 1:],
            ),
            scaled_sizes[..., 0:1] / image_shapes[..., 0:1],
            scaled_sizes[..., 1:] / image_shapes[..., 1:],
        )
        scaled_sizes = tf.round(image_shapes * scales)
        image_scales = scaled_sizes / image_shapes

        # offsets
        max_offsets = scaled_sizes - self.crop_size
        max_offsets = tf.where(
            tf.less(max_offsets, 0), tf.zeros_like(max_offsets), max_offsets
        )
        offsets = max_offsets * self._random_generator.random_uniform(
            shape=(batch_size, 2), minval=0, maxval=1, dtype=tf.float32
        )
        offsets = tf.cast(offsets, dtype=tf.int32)

        # paddings
        new_heights = tf.cast(scaled_sizes[..., 0:1], dtype=tf.int32)
        new_widths = tf.cast(scaled_sizes[..., 1:], dtype=tf.int32)
        tops = tf.where(
            new_heights < self.crop_height,
            tf.cast((self.crop_height - new_heights) / 2, tf.int32),
            0,
        )
        bottoms = tf.where(
            new_heights < self.crop_height,
            self.crop_height - new_heights - tops,
            0,
        )
        lefts = tf.where(
            new_widths < self.crop_width,
            tf.cast((self.crop_width - new_widths) / 2, tf.int32),
            0,
        )
        rights = tf.where(
            new_widths < self.crop_width,
            self.crop_width - new_widths - lefts,
            0,
        )
        (tops, bottoms, lefts, rights) = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, self.position, self._random_generator
        )
        paddings = tf.concat([tops, bottoms, lefts, rights], axis=-1)

        return {
            "image_scales": image_scales,
            "scaled_sizes": scaled_sizes,
            "offsets": offsets,
            "paddings": paddings,
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
        # tf.image.resize always output tf.float32 unless interpolation==nearest
        inputs_for_resize_and_crop_single_image = {
            "images": tf.cast(images, dtype=tf.float32),
            "scaled_sizes": transformations["scaled_sizes"],
            "offsets": transformations["offsets"],
            "paddings": transformations["paddings"],
        }
        images = tf.map_fn(
            self.resize_and_crop_single_image,
            inputs_for_resize_and_crop_single_image,
            fn_output_signature=tf.float32,
        )
        images = tf.ensure_shape(
            images, shape=(None, self.height, self.width, None)
        )
        return tf.cast(images, self.compute_dtype)

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
                "`RandomZoomAndCrop()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomZoomAndCrop(..., bounding_box_format='xyxy')`"
            )
        # cast to float32 to avoid numerical issue
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="yxyx",
            images=raw_images,
            dtype=tf.float32,
        )

        image_scales = tf.cast(
            transformations["image_scales"], dtype=tf.float32
        )
        offsets = tf.cast(transformations["offsets"], dtype=tf.float32)
        paddings = tf.cast(transformations["paddings"], dtype=tf.float32)
        padding_offsets = tf.concat(
            [paddings[..., 0:1], paddings[..., 2:3]], axis=-1
        )

        # Adjusts box coordinates based on image_scale and offset.
        bounding_boxes = bounding_boxes.copy()
        yxyx = bounding_boxes["boxes"]
        yxyx *= tf.tile(image_scales, [1, 2])[..., tf.newaxis, :]
        yxyx -= tf.tile(offsets, [1, 2])[..., tf.newaxis, :]
        yxyx += tf.tile(padding_offsets, [1, 2])[..., tf.newaxis, :]

        bounding_boxes["boxes"] = yxyx
        bounding_boxes = bounding_box_utils.clip_to_image(
            bounding_boxes,
            bounding_box_format="yxyx",
            images=images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="yxyx",
            target=self.bounding_box_format,
            images=images,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    def compute_ragged_segmentation_mask_signature(self, segmentation_masks):
        return tf.RaggedTensorSpec(
            shape=(self.height, self.width, segmentation_masks.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )

    def augment_ragged_segmentation_mask(
        self, segmentation_mask, transformation, **kwargs
    ):
        segmentation_mask = tf.expand_dims(segmentation_mask, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        segmentation_mask = self.augment_segmentation_masks(
            segmentation_masks=segmentation_mask,
            transformations=transformation,
            **kwargs,
        )
        return tf.squeeze(segmentation_mask, axis=0)

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        # unpackage augmentation arguments
        inputs_for_resize_and_crop_single_segmentation_mask = {
            "segmentation_masks": segmentation_masks,
            "scaled_sizes": transformations["scaled_sizes"],
            "offsets": transformations["offsets"],
            "paddings": transformations["paddings"],
        }
        segmentation_masks = tf.map_fn(
            self.resize_and_crop_single_segmentation_mask,
            inputs_for_resize_and_crop_single_segmentation_mask,
            fn_output_signature=segmentation_masks.dtype,
        )
        segmentation_masks = tf.ensure_shape(
            segmentation_masks, shape=(None, self.height, self.width, None)
        )
        return tf.cast(segmentation_masks, self.compute_dtype)

    def resize_and_crop_single_image(self, inputs):
        image = inputs.get("images", None)
        scaled_size = inputs.get("scaled_sizes", None)
        offset = inputs.get("offsets", None)
        padding = inputs.get("paddings", None)

        image = tf.image.resize(
            image,
            tf.cast(scaled_size, tf.int32),
            method=self.interpolation,
            antialias=self.antialias,
        )
        image = image[
            offset[0] : offset[0] + self.crop_height,
            offset[1] : offset[1] + self.crop_width,
            :,
        ]
        paddings = tf.stack(
            (
                tf.stack((padding[0], padding[1])),
                tf.stack((padding[2], padding[3])),
                tf.zeros(shape=(2,), dtype=tf.int32),
            )
        )
        image = tf.pad(
            image, paddings=paddings, constant_values=self.padding_value
        )
        return image

    def resize_and_crop_single_segmentation_mask(self, inputs):
        segmentation_mask = inputs.get("segmentation_masks", None)
        scaled_size = inputs.get("scaled_sizes", None)
        offset = inputs.get("offsets", None)
        padding = inputs.get("paddings", None)

        segmentation_mask = tf.image.resize(
            segmentation_mask, tf.cast(scaled_size, tf.int32), method="nearest"
        )
        segmentation_mask = segmentation_mask[
            offset[0] : offset[0] + self.crop_height,
            offset[1] : offset[1] + self.crop_width,
            :,
        ]
        paddings = tf.stack(
            (
                tf.stack((padding[0], padding[1])),
                tf.stack((padding[2], padding[3])),
                tf.zeros(shape=(2,), dtype=tf.int32),
            )
        )
        segmentation_mask = tf.pad(
            segmentation_mask,
            paddings=paddings,
            constant_values=self.padding_value,
        )
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "scale_factor": self.scale_factor,
                "crop_height": self.crop_height,
                "crop_width": self.crop_width,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config
