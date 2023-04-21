from functools import partial

import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomHSV(VectorizedBaseRandomLayer):
    """Randomly adjusts the hue, saturation and value on given images.

    This layer will randomly increase/reduce the hue, saturation and value for
    the input RGB images.

    The image hue, saturation and value is adjusted by converting the image(s)
    to HSV and rotating the hue channel (H) by hue factor, multiplying the
    saturation channel (S) by saturation factor and multiplying the value
    channel (V) by value factor. The image is then converted back to RGB.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high]. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        hue_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. When represented as a single float,
            lower = upper. The hue factor will be randomly picked between
            `[0.5 - lower, 0.5 + upper]`. 0.0 means no shift, while a value of
            -0.5 or +0.5 gives an image with complementary colors.
        saturation_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. When represented as a single float,
            lower = upper. The saturation factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. 1.0 will give the original
            image, 0.0 makes the image to be fully grayscale while 2.0 will
            enhance the saturation by a factor of 2.
        value_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. When represented as a single float,
            lower = upper. The value factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. 1.0 will give the original
            image, 0.0 makes the image to be zero values while 2.0 will
            enhance the value by a factor of 2.
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        value_range,
        hue_factor,
        saturation_factor,
        value_factor,
        seed=None,
        **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.hue_factor = augmentation_utils.parse_factor(
            hue_factor, min_value=-0.5, max_value=0.5, center_value=0, seed=seed
        )
        self.saturation_factor = augmentation_utils.parse_factor(
            saturation_factor, max_value=None, center_value=1, seed=seed
        )
        self.value_factor = augmentation_utils.parse_factor(
            value_factor, max_value=None, center_value=1, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

        # decide whether to enable the augmentation
        self._enable_hue = augmentation_utils.is_factor_working(
            self.hue_factor, not_working_value=0.0
        )
        self._enable_saturation = augmentation_utils.is_factor_working(
            self.saturation_factor, not_working_value=1.0
        )
        self._enable_value = augmentation_utils.is_factor_working(
            self.value_factor, not_working_value=1.0
        )

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # orders determine the augmentation order which is the same across
        # single batch
        orders = tf.argsort(
            self._random_generator.random_uniform((3,)), axis=-1
        )
        orders = tf.reshape(
            tf.tile(orders, multiples=(batch_size,)), shape=(batch_size, 3)
        )

        hue_factors = self.hue_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        saturation_factors = self.saturation_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        value_factors = self.value_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        return {
            "orders": orders,
            "hue_factors": hue_factors,
            "saturation_factors": saturation_factors,
            "value_factors": value_factors,
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
                    0: partial(self.adjust_hue, images, transformations),
                    1: partial(self.adjust_saturation, images, transformations),
                    2: partial(self.adjust_value, images, transformations),
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

    def adjust_hue(self, images, transformations):
        if not self._enable_hue:
            return images
        # The output is only well defined if the value in images are in [0,1].
        # https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_hsv
        images = preprocessing_utils.transform_value_range(
            images, (0, 255), (0, 1), dtype=self.compute_dtype
        )
        images = tf.image.rgb_to_hsv(images)
        # adjust hue
        hue_factors = transformations["hue_factors"]
        hue_factors = hue_factors[..., tf.newaxis]
        h_channels = images[..., 0] + hue_factors
        h_channels = tf.where(h_channels > 1.0, h_channels - 1.0, h_channels)
        h_channels = tf.where(h_channels < 0.0, h_channels + 1.0, h_channels)
        images = tf.stack([h_channels, images[..., 1], images[..., 2]], axis=-1)
        images = tf.image.hsv_to_rgb(images)
        images = preprocessing_utils.transform_value_range(
            images, (0, 1), (0, 255), dtype=self.compute_dtype
        )
        return images

    def adjust_saturation(self, images, transformations):
        if not self._enable_saturation:
            return images
        saturation_factors = transformations["saturation_factors"]
        saturation_factors = saturation_factors[..., tf.newaxis, tf.newaxis]
        means = tf.image.rgb_to_grayscale(images)
        means = tf.image.grayscale_to_rgb(means)
        images = augmentation_utils.blend(images, means, saturation_factors)
        images = tf.clip_by_value(images, 0, 255)
        return images

    def adjust_value(self, images, transformations):
        if not self._enable_value:
            return images
        value_factors = transformations["value_factors"]
        value_factors = value_factors[..., tf.newaxis, tf.newaxis]
        images = augmentation_utils.blend(
            images, tf.zeros_like(images), value_factors
        )
        images = tf.clip_by_value(images, 0, 255)
        return images

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "hue_factor": self.hue_factor,
                "saturation_factor": self.saturation_factor,
                "value_factor": self.value_factor,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
