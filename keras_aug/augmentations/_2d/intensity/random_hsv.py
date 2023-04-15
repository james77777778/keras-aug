import tensorflow as tf
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomHSV(VectorizedBaseImageAugmentationLayer):
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
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image hue is impacted. `factor=0.0` makes this layer perform a
            no-op operation, while a value of 1.0 performs the most aggressive
            contrast adjustment available. If a tuple is used, a `factor` is
            sampled between the two values for every image augmented. If a
            single float is used, a value between `0.0` and the passed float is
            sampled. In order to ensure the value is always the same, please
            pass a tuple with two identical floats: `(0.5, 0.5)`.
        saturation_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image saturation is impacted. `factor=0.5` makes this layer perform
            a no-op operation. `factor=0.0` makes the image to be fully
            grayscale. `factor=1.0` makes the image to be fully saturated.
            Values should be between `0.0` and `1.0`. If a tuple is used, a
            `factor` is sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the
            passed float is sampled. In order to ensure the value is always the
            same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
        value_factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. `factor` controls the extent to which the
            image saturation is impacted. `factor=0.5` makes this layer perform
            a no-op operation. `factor=0.0` makes the image to be fully
            grayscale. `factor=1.0` makes the image to be fully valued.
            Values should be between `0.0` and `1.0`. If a tuple is used, a
            `factor` is sampled between the two values for every image
            augmented. If a single float is used, a value between `0.0` and the
            passed float is sampled. In order to ensure the value is always the
            same, please pass a tuple with two identical floats: `(0.5, 0.5)`.
        seed: Integer. Used to create a random seed.
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
        self.hue_factor = preprocessing_utils.parse_factor(
            hue_factor, seed=seed
        )
        self.saturation_factor = preprocessing_utils.parse_factor(
            saturation_factor, seed=seed
        )
        self.value_factor = preprocessing_utils.parse_factor(
            value_factor, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # We must scale self.hue_factor() to the range [-0.5, 0.5]. This is
        # because the operation performs rotation on the hue saturation value
        # orientation. This can be thought of as an angle in the range
        # [-180, 180]
        hue_factors = -0.5 + self.hue_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )

        # Convert the self.saturation_factor and self.value_factor range from
        # [0, 1] to [0, +inf].
        # The adjustments are trying to apply the following math
        # formula `outputs = inputs * factor`. We use the following method to
        # the do the mapping: `y = x / (1 - x)`.
        # This will ensure:
        #   y = +inf when x = 1 (full saturation / full value)
        #   y = 1 when x = 0.5 (no augmentation)
        #   y = 0 when x = 0 (full grayscale / zero value)
        saturation_factors = self.saturation_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        saturation_factors = saturation_factors / (1.0 - saturation_factors)
        value_factors = self.value_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        value_factors = value_factors / (1.0 - value_factors)
        return {
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
            images, self.value_range, (0, 1), dtype=self.compute_dtype
        )
        hue_factors = transformations["hue_factors"]
        saturation_factors = transformations["saturation_factors"]
        value_factors = transformations["value_factors"]

        # The output is only well defined if the value in images are in [0,1].
        # https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_hsv
        images = tf.image.rgb_to_hsv(images)

        # adjust hue
        h_channels = images[..., 0] + hue_factors[..., tf.newaxis]
        h_channels = tf.where(h_channels > 1.0, h_channels - 1.0, h_channels)
        h_channels = tf.where(h_channels < 0.0, h_channels + 1.0, h_channels)

        # adjust saturation
        s_channels = tf.multiply(
            images[..., 1], saturation_factors[..., tf.newaxis]
        )
        s_channels = tf.clip_by_value(
            s_channels, clip_value_min=0.0, clip_value_max=1.0
        )

        # adjust value
        v_channels = tf.multiply(images[..., 2], value_factors[..., tf.newaxis])
        v_channels = tf.clip_by_value(
            v_channels, clip_value_min=0.0, clip_value_max=1.0
        )

        # stack hues, saturations, values back to images
        images = tf.stack([h_channels, s_channels, v_channels], axis=-1)
        images = tf.image.hsv_to_rgb(images)
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
