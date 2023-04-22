from functools import partial

import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomBrightnessContrast(VectorizedBaseRandomLayer):
    """RandomBrightnessContrast class randomly apply brightness and contrast
    image processing operation sequentially and randomly on the input.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is set up.
        brightness_factor:  A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. When represented as a single float,
            lower = upper. The brightness factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. 0.0 will make image be black. 1.0 will
            make image be white.
        contrast_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. When represented as a single float,
            lower = upper. The contrast factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. 0.0 gives solid gray image, 1.0 gives
            the original image while 2.0 increases the contrast by a factor of
            2.
        seed: Used to create a random seed, defaults to None.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(
        self,
        value_range,
        brightness_factor,
        contrast_factor,
        seed=None,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.brightness_factor = augmentation_utils.parse_factor(
            brightness_factor, max_value=None, center_value=1, seed=seed
        )
        self.contrast_factor = augmentation_utils.parse_factor(
            contrast_factor, max_value=None, center_value=1, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

        # decide whether to enable the augmentation
        self._enable_brightness = augmentation_utils.is_factor_working(
            self.brightness_factor, not_working_value=1.0
        )
        self._enable_contrast = augmentation_utils.is_factor_working(
            self.contrast_factor, not_working_value=1.0
        )

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # orders determine the augmentation order which is the same across
        # single batch
        orders = tf.argsort(
            self._random_generator.random_uniform((2,)), axis=-1
        )
        orders = tf.reshape(
            tf.tile(orders, multiples=(batch_size,)), shape=(batch_size, 2)
        )

        brightness_factors = self.brightness_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        contrast_factors = self.contrast_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )

        return {
            "orders": orders,
            "brightness_factors": brightness_factors,
            "contrast_factors": contrast_factors,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        transformations = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        images = self.augment_images(
            images=images, transformations=transformations, **kwargs
        )
        return tf.squeeze(images, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, (0, 255), dtype=self.compute_dtype
        )
        orders = transformations["orders"][0]
        for order in orders:
            images = tf.switch_case(
                order,
                branch_fns={
                    0: partial(self.adjust_brightness, images, transformations),
                    1: partial(self.adjust_contrast, images, transformations),
                },
            )
        images = preprocessing_utils.transform_value_range(
            images, (0, 255), self.value_range, dtype=self.compute_dtype
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

    def adjust_brightness(self, images, transformations):
        if not self._enable_brightness:
            return images
        brightness_factors = transformations["brightness_factors"]
        brightness_factors = brightness_factors[..., tf.newaxis, tf.newaxis]
        images = augmentation_utils.blend(
            images, tf.zeros_like(images), brightness_factors
        )
        images = tf.clip_by_value(images, 0, 255)
        return images

    def adjust_contrast(self, images, transformations):
        if not self._enable_contrast:
            return images
        contrast_factors = transformations["contrast_factors"]
        contrast_factors = contrast_factors[..., tf.newaxis, tf.newaxis]
        means = tf.image.rgb_to_grayscale(images)
        means = tf.image.grayscale_to_rgb(means)
        images = augmentation_utils.blend(images, means, contrast_factors)
        images = tf.clip_by_value(images, 0, 255)
        return images

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "brightness_factor": self.brightness_factor,
                "contrast_factor": self.contrast_factor,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
