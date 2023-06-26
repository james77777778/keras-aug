import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug import core
from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils import bounding_box as bounding_box_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomCropAndResize(VectorizedBaseRandomLayer):
    """Randomly crops a part of an image and resizes it to provided size.

    This implementation takes an intuitive approach, where we crop the images to
    a random height and width, and then resize them. To do this, we first sample
    a random value for area using `crop_area_factor` and a value for aspect
    ratio using `aspect_ratio_factor`. Further we get the new height and width
    by dividing and multiplying the old height and width by the random area
    respectively. We then sample offsets for height and width and clip them such
    that the cropped area does not exceed image boundaries. Finally, we do the
    actual cropping operation and resize the image to (`height`, `width`).

    Args:
        height (int): The height of result image.
        width (int): The width of result image.
        crop_area_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range of the area of the cropped part to that of original image.
            For self-supervised pretraining a common value for this parameter is
            ``(0.08, 1.0)``. For fine-tuning and classification a common value
            is ``(0.8, 1.0)``.
        aspect_ratio_factor (float|Sequence[float]|keras_aug.FactorSampler): The
            ratio of width to height of the cropped image. When represented as
            a single float, the factor will be picked between
            ``[1.0 - factor, 1.0]``. For most tasks, this should be
            ``(3/4, 4/3)``.
        interpolation (str, optional): The interpolation mode. Supported values:
            ``"nearest", "bilinear"``. Defaults to `"bilinear"`.
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
        crop_area_factor,
        aspect_ratio_factor,
        interpolation="bilinear",
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self._check_arguments(
            height, width, crop_area_factor, aspect_ratio_factor
        )
        self.height = height
        self.width = width
        self.aspect_ratio_factor = augmentation_utils.parse_factor(
            aspect_ratio_factor,
            min_value=0.0,
            max_value=None,
            center_value=1.0,
            seed=seed,
        )
        if isinstance(crop_area_factor, float):
            lower = 1.0 - crop_area_factor
            upper = 1.0
            crop_area_factor = (lower, upper)
        self.crop_area_factor = augmentation_utils.parse_factor(
            crop_area_factor,
            min_value=0.0,
            max_value=1.0,
            seed=seed,
        )
        self.interpolation = interpolation
        self.bounding_box_format = bounding_box_format
        self.seed = seed

        # set force_output_dense_images=True because the output images must
        # have same shape (B, height, width, C)
        self.force_output_dense_images = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        crop_area_factor = self.crop_area_factor(shape=(batch_size, 1))
        aspect_ratio = self.aspect_ratio_factor(shape=(batch_size, 1))

        new_height = tf.clip_by_value(
            tf.sqrt(crop_area_factor / aspect_ratio), 0.0, 1.0
        )  # to avoid unwanted/unintuitive effects
        new_width = tf.clip_by_value(
            tf.sqrt(crop_area_factor * aspect_ratio), 0.0, 1.0
        )

        height_offset = self._random_generator.random_uniform(
            shape=(batch_size, 1),
            minval=tf.minimum(0.0, 1.0 - new_height),
            maxval=tf.maximum(0.0, 1.0 - new_height),
            dtype=tf.float32,
        )

        width_offset = self._random_generator.random_uniform(
            shape=(batch_size, 1),
            minval=tf.minimum(0.0, 1.0 - new_width),
            maxval=tf.maximum(0.0, 1.0 - new_width),
            dtype=tf.float32,
        )

        y1s = height_offset
        y2s = height_offset + new_height
        x1s = width_offset
        x2s = width_offset + new_width

        boxes = tf.concat([y1s, x1s, y2s, x2s], axis=-1)
        return boxes

    def compute_ragged_image_signature(self, images):
        return tf.RaggedTensorSpec(
            shape=(self.height, self.width, images.shape[-1]),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        batch_size = tf.shape(images)[0]
        boxes = transformations
        indices = tf.range(batch_size)

        # tf.image.crop_and_resize not support bfloat16
        if images.dtype == tf.bfloat16:
            images = tf.cast(images, dtype=tf.float32)
        images = tf.image.crop_and_resize(
            images,  # image shape: [B, H, W, C]
            boxes,  # boxes: (B, 4) in this case; represents area
            # to be cropped from the original image
            indices,  # box_indices: maps boxes to images along batch axis
            [self.height, self.width],  # output size
            method=self.interpolation,
        )
        images = tf.ensure_shape(
            images, shape=(None, self.height, self.width, None)
        )
        return tf.cast(images, dtype=self.compute_dtype)

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
                "`RandomCropAndResize()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCropAndResize(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            images=raw_images,
        )

        t_y1s, t_x1s, t_y2s, t_x2s = tf.split(transformations, 4, axis=-1)
        # broadcast
        t_y1s, t_x1s = t_y1s[..., tf.newaxis], t_x1s[..., tf.newaxis]
        t_y2s, t_x2s = t_y2s[..., tf.newaxis], t_x2s[..., tf.newaxis]

        t_dxs = t_x2s - t_x1s
        t_dys = t_y2s - t_y1s
        x1s, y1s, x2s, y2s = tf.split(bounding_boxes["boxes"], 4, axis=-1)
        output = tf.concat(
            [
                (x1s - t_x1s) / t_dxs,
                (y1s - t_y1s) / t_dys,
                (x2s - t_x1s) / t_dxs,
                (y2s - t_y1s) / t_dys,
            ],
            axis=-1,
        )
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = output

        bounding_boxes = bounding_box_utils.clip_to_image(
            bounding_boxes,
            bounding_box_format="rel_xyxy",
            images=images,
        )
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=images,
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
        transformation = tf.expand_dims(transformation, axis=0)
        segmentation_mask = self.augment_segmentation_masks(
            segmentation_masks=segmentation_mask,
            transformations=transformation,
            **kwargs,
        )
        return tf.squeeze(segmentation_mask, axis=0)

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        batch_size = tf.shape(segmentation_masks)[0]
        boxes = transformations
        indices = tf.range(batch_size)

        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if segmentation_masks.dtype == tf.bfloat16:
            segmentation_masks = tf.cast(segmentation_masks, dtype=tf.float32)
        segmentation_masks = tf.image.crop_and_resize(
            segmentation_masks,  # image shape: [B, H, W, C]
            boxes,  # boxes: (B, 4) in this case; represents area
            # to be cropped from the original image
            indices,  # box_indices: maps boxes to images along batch axis
            [self.height, self.width],  # output size
            method="nearest",
        )
        segmentation_masks = tf.ensure_shape(
            segmentation_masks, shape=(None, self.height, self.width, None)
        )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

    def _check_arguments(
        self, height, width, crop_area_factor, aspect_ratio_factor
    ):
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(
                "`height` and `width` must be integer. Received: "
                f"`height`={height} `width`={width} "
            )

        if (
            not isinstance(crop_area_factor, (tuple, list, core.FactorSampler))
            or isinstance(crop_area_factor, float)
            or isinstance(crop_area_factor, int)
        ):
            raise ValueError(
                "`crop_area_factor` must be tuple of two positive floats less "
                "than or equal to 1 or keras_cv.core.FactorSampler instance. "
                f"Received crop_area_factor={crop_area_factor}"
            )

        if (
            not isinstance(
                aspect_ratio_factor, (tuple, list, core.FactorSampler)
            )
            or isinstance(aspect_ratio_factor, float)
            or isinstance(aspect_ratio_factor, int)
        ):
            raise ValueError(
                "`aspect_ratio_factor` must be tuple of two positive floats or "
                "keras_cv.core.FactorSampler instance. Received "
                f"aspect_ratio_factor={aspect_ratio_factor}"
            )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "crop_area_factor": self.crop_area_factor,
                "aspect_ratio_factor": self.aspect_ratio_factor,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config
