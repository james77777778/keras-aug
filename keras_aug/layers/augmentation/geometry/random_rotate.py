import math

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
class RandomRotate(VectorizedBaseRandomLayer):
    """Randomly rotates the input images.

    The unit of the factor is degree. A positive value means rotating counter
    clock-wise, while a negative value means clock-wise.

    Args:
        factor (float|Sequence[float]|keras_aug.FactorSampler): The
            range of the degree for random rotation. When represented as a
            single float, the factor will be picked between
            ``[0.0 - lower, 0.0 + upper]``. A positive value means rotating
            counter clock-wise, while a negative value means clock-wise.
        interpolation (str, optional): The interpolation mode. Supported values:
            ``"nearest", "bilinear"``. Defaults to `"bilinear"`.
        fill_mode (str, optional): The fill mode. Supported values:
            ``"constant", "reflect", "wrap", "nearest"``. Defaults to
            ``"constant"``.
        fill_value (int|float, optional): The value to be filled outside the
            boundaries when ``fill_mode="constant"``. Defaults to ``0``.
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
        factor,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.factor = augmentation_utils.parse_factor(
            factor,
            min_value=-180,
            max_value=180,
            center_value=0,
            seed=seed,
        )

        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        # cast to float32 to avoid numerical issue
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        angles = self.factor(shape=(batch_size, 1), dtype=tf.float32)
        angles = angles / 360.0 * 2.0 * math.pi
        rotation_matrixes = augmentation_utils.get_rotation_matrix(
            angles, heights, widths, to_square=True
        )
        # (batch_size, 3, 3)
        return {
            "angles": angles,
            "rotation_matrixes": rotation_matrixes,
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
        original_shape = images.shape
        batch_size = tf.shape(images)[0]
        rotation_matrixes = transformations["rotation_matrixes"]
        rotation_matrixes = tf.reshape(
            rotation_matrixes, shape=(batch_size, -1)
        )
        rotation_matrixes = rotation_matrixes[:, :-1]

        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if images.dtype == tf.bfloat16:
            images = tf.cast(images, dtype=tf.float32)
        images = preprocessing_utils.transform(
            images,
            rotation_matrixes,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        images = tf.ensure_shape(images, shape=original_shape)
        return tf.cast(images, dtype=self.compute_dtype)

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomAffine()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomAffine(bounding_box_format='xyxy')`"
            )
        # cast to float32 to avoid numerical issue
        heights, widths = augmentation_utils.get_images_shape(
            raw_images, dtype=tf.float32
        )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
            dtype=tf.float32,
        )
        boxes = bounding_boxes["boxes"]

        # process rotations
        origin_x = widths / 2
        origin_y = heights / 2
        angles = -transformations["angles"]
        angles = angles[:, tf.newaxis, tf.newaxis]
        # points: (batch_size, max_num_boxes, 4, 2)
        points = tf.stack(
            [
                tf.stack([boxes[:, :, 0], boxes[:, :, 1]], axis=2),
                tf.stack([boxes[:, :, 2], boxes[:, :, 1]], axis=2),
                tf.stack([boxes[:, :, 2], boxes[:, :, 3]], axis=2),
                tf.stack([boxes[:, :, 0], boxes[:, :, 3]], axis=2),
            ],
            axis=2,
        )
        point_x_offsets = points[..., 0] - origin_x[..., tf.newaxis]
        point_y_offsets = points[..., 1] - origin_y[..., tf.newaxis]
        new_x = (
            origin_x[..., tf.newaxis, tf.newaxis]
            + tf.multiply(tf.cos(angles), point_x_offsets[..., tf.newaxis])
            - tf.multiply(tf.sin(angles), point_y_offsets[..., tf.newaxis])
        )
        new_y = (
            origin_y[..., tf.newaxis, tf.newaxis]
            + tf.multiply(tf.sin(angles), point_x_offsets[..., tf.newaxis])
            + tf.multiply(tf.cos(angles), point_y_offsets[..., tf.newaxis])
        )
        out = tf.concat([new_x, new_y], axis=3)
        min_cordinates = tf.math.reduce_min(out, axis=2)
        max_cordinates = tf.math.reduce_max(out, axis=2)
        boxes = tf.concat([min_cordinates, max_cordinates], axis=2)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        bounding_boxes = bounding_box_utils.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=raw_images,
        )
        # coordinates cannot be float values, it is cast to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=raw_images,
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
        original_shape = segmentation_masks.shape
        batch_size = tf.shape(segmentation_masks)[0]
        rotation_matrixes = transformations["rotation_matrixes"]
        rotation_matrixes = tf.reshape(
            rotation_matrixes, shape=(batch_size, -1)
        )
        rotation_matrixes = rotation_matrixes[:, :-1]

        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if segmentation_masks.dtype == tf.bfloat16:
            segmentation_masks = tf.cast(segmentation_masks, dtype=tf.float32)
        segmentation_masks = preprocessing_utils.transform(
            segmentation_masks,
            rotation_matrixes,
            fill_mode=self.fill_mode,
            fill_value=0,
            interpolation="nearest",
        )
        segmentation_masks = tf.ensure_shape(
            segmentation_masks, shape=original_shape
        )
        return tf.cast(segmentation_masks, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config
