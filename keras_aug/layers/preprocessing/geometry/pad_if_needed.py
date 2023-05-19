import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class PadIfNeeded(VectorizedBaseRandomLayer):
    """Pads the images if needed.

    PadIfNeeded can be configured by specifying the height/width or the
    divisors. The images will be padded to ``(min_height, min_width)`` or the
    size of the both sides to be divisible by ``height_divisor`` and
    ``width_divisor``. PadIfNeeded is required to specify
    ``min_height`` or ``height_divisor`` and ``min_width`` or
    ``width_divisor``, respectively.

    Args:
        min_height (int, optional): The height of result image.
        min_width (int, optional): The width of result image.
        height_divisor (int, optional): The divisor that ensures image height is
            divisible by.
        width_divisor (int, optional): The divisor that ensures image width is
            divisible by.
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
        - `Albumentations <https://github.com/albumentations-team/albumentations>`_
    """  # noqa: E501

    def __init__(
        self,
        min_height=None,
        min_width=None,
        pad_height_divisor=None,
        pad_width_divisor=None,
        position="center",
        padding_value=0,
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
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(images)

        if self.min_height is not None:
            tops = tf.where(
                heights < self.min_height,
                tf.cast((self.min_height - heights) / 2, heights.dtype),
                tf.zeros_like(heights, dtype=heights.dtype),
            )
            bottoms = tf.where(
                heights < self.min_height,
                self.min_height - heights - tops,
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
            tops = tf.round(pad_rows / 2.0)
            bottoms = tf.cast(pad_rows - tops, dtype=tf.int32)
            tops = tf.cast(tops, dtype=tf.int32)

        if self.min_width is not None:
            lefts = tf.where(
                widths < self.min_width,
                tf.cast((self.min_width - widths) / 2, widths.dtype),
                tf.zeros_like(widths, dtype=widths.dtype),
            )
            rights = tf.where(
                widths < self.min_width,
                self.min_width - widths - lefts,
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
            lefts = tf.round(pad_cols / 2.0)
            rights = tf.cast(pad_cols - lefts, dtype=tf.int32)
            lefts = tf.cast(lefts, dtype=tf.int32)

        (tops, bottoms, lefts, rights) = augmentation_utils.get_position_params(
            tops, bottoms, lefts, rights, self.position, self._random_generator
        )

        return {
            "pad_tops": tops,
            "pad_bottoms": bottoms,
            "pad_lefts": lefts,
            "pad_rights": rights,
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
        images = tf.pad(
            images, paddings=paddings, constant_values=self.padding_value
        )
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
        segmentation_masks = tf.pad(
            segmentation_masks, paddings=paddings, constant_values=0
        )
        return segmentation_masks

    def augment_segmentation_mask_single(self, inputs):
        segmentation_mask = inputs.get(
            augmentation_utils.SEGMENTATION_MASKS, None
        )
        transformation = inputs.get("transformation", None)
        pad_top = transformation["pad_tops"][0]
        pad_bottom = transformation["pad_bottoms"][0]
        pad_left = transformation["pad_lefts"][0]
        pad_right = transformation["pad_rights"][0]
        paddings = tf.stack(
            (
                tf.stack((pad_top, pad_bottom)),
                tf.stack((pad_left, pad_right)),
                tf.zeros(shape=(2,), dtype=pad_top.dtype),
            )
        )
        segmentation_mask = tf.pad(
            segmentation_mask, paddings=paddings, constant_values=0
        )
        return segmentation_mask

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "min_height": self.min_height,
                "min_width": self.min_width,
                "pad_height_divisor": self.pad_height_divisor,
                "pad_width_divisor": self.pad_width_divisor,
                "position": self.position,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config
