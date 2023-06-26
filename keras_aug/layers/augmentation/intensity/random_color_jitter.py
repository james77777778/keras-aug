import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomColorJitter(VectorizedBaseRandomLayer):
    """Randomly applies brightness, contrast, saturation and hue image
    processing operation sequentially and randomly on the input images. It
    expects input as RGB image.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        brightness_factor (float|Sequence[float]|keras_aug.FactorSampler, optional): The
            range of the brightness factor. When represented as a single float,
            the factor will be picked between ``[1.0 - lower, 1.0 + upper]``.
            ``0.0`` will make image be black. ``1.0`` will make image be white.
            Defaults to ``None``.
        contrast_factor (float|Sequence[float]|keras_aug.FactorSampler, optional): The range
            of the contrast factor. When represented as a single float, the
            factor will be picked between ``[1.0 - lower, 1.0 + upper]``.
            ``0.0`` gives solid gray image. ``1.0`` gives the original image
            while ``2.0`` increases the contrast by a factor of 2.
            Defaults to ``None``.
        saturation_factor (float|Sequence[float]|keras_aug.FactorSampler, optional): The
            range of the saturation factor. When represented as a single float,
            the factor will be picked between ``[1.0 - lower, 1.0 + upper]``.
            ``1.0`` will give the original image. ``0.0`` makes the image to be
            fully grayscale. ``2.0`` will enhance the saturation by a factor of
            2. Defaults to ``None``.
        hue_factor (float|Sequence[float]|keras_aug.FactorSampler, optional): The range of
            the hue factor. When represented as a single float, the factor will
            be picked between ``[0.0 - lower, 0.0 + upper]``. ``0.0`` means no
            shift. ``-0.5`` or ``0.5`` gives an image with complementary colors.
            Defaults to ``None``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `torchvision <https://github.com/pytorch/vision>`_
        - `Tensorflow Model augment <https://github.com/tensorflow/models/blob/master/official/vision/ops/augment.py>`
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
        brightness_factor=None,
        contrast_factor=None,
        saturation_factor=None,
        hue_factor=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.brightness_factor = None
        self.contrast_factor = None
        self.saturation_factor = None
        self.hue_factor = None
        if brightness_factor:
            self.brightness_factor = augmentation_utils.parse_factor(
                brightness_factor, max_value=None, center_value=1, seed=seed
            )
        if contrast_factor is not None:
            self.contrast_factor = augmentation_utils.parse_factor(
                contrast_factor, max_value=None, center_value=1, seed=seed
            )
        if saturation_factor is not None:
            self.saturation_factor = augmentation_utils.parse_factor(
                saturation_factor, max_value=None, center_value=1, seed=seed
            )
        if hue_factor is not None:
            self.hue_factor = augmentation_utils.parse_factor(
                hue_factor,
                min_value=-0.5,
                max_value=0.5,
                center_value=0,
                seed=seed,
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
        self._enable_saturation = augmentation_utils.is_factor_working(
            self.saturation_factor, not_working_value=1.0
        )
        self._enable_hue = augmentation_utils.is_factor_working(
            self.hue_factor, not_working_value=0.0
        )

    def get_random_transformation_batch(self, batch_size, **kwargs):
        factor_shape = (batch_size, 1, 1, 1)
        # dummy
        brightness_factors = tf.zeros(factor_shape)
        contrast_factors = tf.zeros(factor_shape)
        saturation_factors = tf.zeros(factor_shape)
        hue_factors = tf.zeros(factor_shape)
        if self._enable_brightness:
            brightness_factors = self.brightness_factor(
                shape=factor_shape, dtype=self.compute_dtype
            )
        if self._enable_contrast:
            contrast_factors = self.contrast_factor(
                shape=factor_shape, dtype=self.compute_dtype
            )
        if self._enable_saturation:
            saturation_factors = self.saturation_factor(
                shape=factor_shape, dtype=self.compute_dtype
            )
        if self._enable_hue:
            hue_factors = self.hue_factor(
                shape=factor_shape, dtype=self.compute_dtype
            )
        return {
            "brightness_factors": brightness_factors,
            "contrast_factors": contrast_factors,
            "saturation_factors": saturation_factors,
            "hue_factors": hue_factors,
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
        if self._enable_brightness:
            images = self.adjust_brightness(images, transformations)
        if self._enable_contrast:
            images = self.adjust_contrast(images, transformations)
        if self._enable_saturation:
            images = self.adjust_saturation(images, transformations)
        if self._enable_hue:
            images = self.adjust_hue(images, transformations)
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
        brightness_factors = transformations["brightness_factors"]
        images *= brightness_factors
        images = tf.clip_by_value(images, 0, 255)
        return images

    def adjust_contrast(self, images, transformations):
        # cast to float32 to avoid numerical issue
        contrast_factors = tf.cast(
            transformations["contrast_factors"], dtype=tf.float32
        )
        images = tf.cast(images, dtype=tf.float32)
        degenerates = augmentation_utils.rgb_to_grayscale(images)
        degenerates = tf.reduce_mean(degenerates, axis=(1, 2, 3), keepdims=True)
        images = augmentation_utils.blend(degenerates, images, contrast_factors)
        images = tf.clip_by_value(images, 0, 255)
        return tf.cast(images, dtype=self.compute_dtype)

    def adjust_saturation(self, images, transformations):
        saturation_factors = transformations["saturation_factors"]
        degenerates = augmentation_utils.rgb_to_grayscale(images)
        images = augmentation_utils.blend(
            degenerates, images, saturation_factors, (0, 255)
        )
        images = tf.clip_by_value(images, 0, 255)
        return images

    def adjust_hue(self, images, transformations):
        # cast to float32 to avoid numerical issue
        images = tf.image.rgb_to_hsv(tf.cast(images, dtype=tf.float32))
        hue_factors = tf.cast(transformations["hue_factors"], dtype=tf.float32)
        h_channels = tf.math.floormod(images[..., 0:1] + hue_factors, 1.0)
        images = tf.concat(
            [h_channels, images[..., 1:2], images[..., 2:3]], axis=-1
        )
        images = tf.image.hsv_to_rgb(images)
        images = tf.clip_by_value(images, 0, 255)
        return tf.cast(images, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "brightness_factor": self.brightness_factor,
                "contrast_factor": self.contrast_factor,
                "saturation_factor": self.saturation_factor,
                "hue_factor": self.hue_factor,
                "seed": self.seed,
            }
        )
        return config
