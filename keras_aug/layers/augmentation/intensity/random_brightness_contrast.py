from functools import partial

import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomBrightnessContrast(VectorizedBaseRandomLayer):
    """Randomly applies brightness and contrast image processing operation
    sequentially and randomly on the input. It expects input as RGB image.

    Args:
        value_range ((int|float, int|float)): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        brightness_factor (float|(float, float)|keras_cv.FactorSampler): The
            range of the brightness factor. When represented as a single float,
            the factor will be picked between ``[1.0 - lower, 1.0 + upper]``.
            ``0.0`` will make image be black. ``1.0`` will make image be white.
        contrast_factor (float|(float, float)|keras_cv.FactorSampler): The range
            of the contrast factor. When represented as a single float, the
            factor will be picked between ``[1.0 - lower, 1.0 + upper]``.
            ``0.0`` gives solid gray image. ``1.0`` gives the original image
            while ``2.0`` increases the contrast by a factor of 2.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `Tensorflow Model augment <https://github.com/tensorflow/models/blob/master/official/vision/ops/augment.py>`
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

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
        # order determines the augmentation order which is the same across
        # single batch
        order = tf.argsort(self._random_generator.random_uniform((2,)), axis=-1)
        order = tf.reshape(
            tf.tile(order, multiples=(batch_size,)), shape=(batch_size, 2)
        )

        brightness_factors = self.brightness_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        contrast_factors = self.contrast_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )

        return {
            "order": order,
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
        order = transformations["order"][0]
        for idx in order:
            images = tf.switch_case(
                idx,
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
            tf.zeros_like(images), images, brightness_factors, (0, 255)
        )
        return images

    def adjust_contrast(self, images, transformations):
        if not self._enable_contrast:
            return images
        batch_size = tf.shape(images)[0]
        contrast_factors = transformations["contrast_factors"]
        contrast_factors = contrast_factors[..., tf.newaxis, tf.newaxis]
        degenerates = tf.image.rgb_to_grayscale(images)

        # compute historams
        degenerates = tf.cast(degenerates, dtype=tf.int32)
        degenerates = tf.reshape(degenerates, shape=(batch_size, -1))
        degenerates = tf.transpose(degenerates, (1, 0))
        hists = tf.math.reduce_sum(
            tf.one_hot(degenerates, depth=256, on_value=1, off_value=0, axis=0),
            axis=1,
        )
        hists = tf.transpose(hists, (1, 0))

        # compute means of historams
        means = tf.reduce_mean(tf.cast(hists, dtype=self.compute_dtype), axis=1)
        means = means / 256.0
        means = means[..., tf.newaxis, tf.newaxis, tf.newaxis]
        degenerates = tf.ones_like(images, dtype=self.compute_dtype) * means
        degenerates = tf.clip_by_value(degenerates, 0, 255)

        images = augmentation_utils.blend(
            degenerates, images, contrast_factors, (0, 255)
        )
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