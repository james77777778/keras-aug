from typing import Sequence

import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug import core
from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomJpegQuality(VectorizedBaseRandomLayer):
    """Randomly applies jpeg compression artifacts to the input images.

    Performs the jpeg compression algorithm on the image. This layer can be used
    in order to ensure your model is robust to artifacts introduced by JPEG
    compression.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        factor (int|Sequence[int]|keras_aug.FactorSampler): The range of the
            compression factor. When represented as a single int, the
            factor will be randomly picked between ``[100 - factor, 100]``.
            ``50`` will give the image with 50% JPEG compression. ``100`` will
            still give a lossy compresson. This value is passed to
            ``tf.image.adjust_jpeg_quality()``.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(
        self,
        value_range,
        factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(factor, (int, float)):
            lower = 101 - int(factor)
            upper = 101
            factor = (lower, upper)
        elif isinstance(factor, Sequence):
            factor = (factor[0], factor[1] + 1)
        elif isinstance(factor, core.FactorSampler):
            factor = factor
        else:
            raise ValueError(
                "RandomJpegQuality expects factor to be a list or a tuple of "
                f"ints. Got factor = {factor}"
            )
        self.factor = augmentation_utils.parse_factor(
            factor, min_value=0, max_value=100 + 1, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # scale from [0, 1] to [0, 100]
        factors = self.factor(shape=(batch_size, 1), dtype=tf.int32)
        return factors

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, target_range=(0, 1)
        )
        inputs_for_adjust_jpeg_qualitye = {
            "images": images,
            "factors": transformations,
        }
        images = tf.vectorized_map(
            self.adjust_jpeg_quality,
            inputs_for_adjust_jpeg_qualitye,
        )
        images = preprocessing_utils.transform_value_range(
            images, (0, 1), self.value_range, dtype=self.compute_dtype
        )
        return images

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

    def adjust_jpeg_quality(self, inputs):
        image = inputs.get("images", None)

        image = tf.cast(image, tf.float32)
        factor = inputs.get("factors", None)

        image = tf.image.adjust_jpeg_quality(image, jpeg_quality=factor[0])
        return image

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "factor": self.factor,
                "seed": self.seed,
            }
        )
        return config
