import tensorflow as tf
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomBrightnessContrast(VectorizedBaseImageAugmentationLayer):
    """RandomBrightnessContrast class randomly apply brightness and contrast
    image processing operation sequentially and randomly on the input.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `channels_last` format

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high].
            This is typically either `[0, 1]` or `[0, 255]` depending
            on how your preprocessing pipeline is set up.
        brightness_factor:  positive float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound. When
            represented as a single float, lower = upper. The brightness factor
            will be randomly picked between `[0.5 - lower, 0.5 + upper]`. When
            0.0 is chosen, the output image will be black, and when 1.0 is
            chosen, the image will be fully white.
        contrast_factor: A positive float represented as fraction of value,
            or a tuple of size 2 representing lower and upper bound. When
            represented as a single float, lower = upper. The contrast factor
            will be randomly picked between `[0.5 - lower, 0.5 + upper]`.
        seed: Integer. Used to create a random seed.
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
        self.brightness_factor = preprocessing_utils.parse_factor(
            brightness_factor, seed=seed
        )
        self.contrast_factor = preprocessing_utils.parse_factor(
            contrast_factor, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # scale self.brightness_factors() from [0, 1] to [-1, 1]
        brightness_factors = self.brightness_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        brightness_factors = brightness_factors * 2.0 - 1.0

        # scale self.contrast_factors() from [0, 1] to [0, +inf]
        # This will ensure:
        #   y = +inf when x = 1 (max contrast)
        #   y = 1 when x = 0.5 (no augmentation)
        #   y = 0 when x = 0 (min contrast)
        contrast_factors = self.contrast_factor(
            shape=(batch_size, 1), dtype=self.compute_dtype
        )
        contrast_factors = contrast_factors / (1.0 - contrast_factors)

        return {
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
        # adjust contrast (must be first augmentation due to mean operation)
        contrast_factors = transformations["contrast_factors"]
        contrast_factors = contrast_factors[..., tf.newaxis, tf.newaxis]
        means = tf.reduce_mean(images, axis=(1, 2), keepdims=True)
        images = (images - means) * contrast_factors + means
        images = tf.clip_by_value(
            images, self.value_range[0], self.value_range[1]
        )

        # adjust brightness
        brightness_factors = transformations["brightness_factors"]
        brightness_factors = brightness_factors[..., tf.newaxis, tf.newaxis]
        brightness_factors *= self.value_range[1] - self.value_range[0]
        images += brightness_factors
        images = tf.clip_by_value(
            images, self.value_range[0], self.value_range[1]
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
                "brightness_factor": self.brightness_factor,
                "contrast_factor": self.contrast_factor,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
