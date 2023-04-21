import tensorflow as tf
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class Normalize(VectorizedBaseRandomLayer):
    """Normalizes the mean and std on given images.

    Args:
        value_range: the range of values the incoming images will have.
            Represented as a two number tuple written [low, high]. This is
            typically either `[0, 1]` or `[0, 255]` depending on how your
            preprocessing pipeline is set up.
        mean: A tuple of int represents the mean values, defaults to
            `(0.485, 0.456, 0.406)` which is the mean values from ImageNet
        std: A tuple of int represents the std values, defaults to
            `(0.229, 0.224, 0.225)` which is the std values from ImageNet
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        value_range,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.mean_input = mean
        self.std_input = std

        num_channel = len(mean)
        self.mean = tf.convert_to_tensor(mean, dtype=self.compute_dtype)
        self.mean = tf.reshape(self.mean, shape=(1, 1, 1, num_channel))
        self.std = tf.convert_to_tensor(std, dtype=self.compute_dtype)
        self.std = tf.reshape(self.std, shape=(1, 1, 1, num_channel))

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        images = self.augment_images(
            images=images, transformations=transformation, **kwargs
        )
        return tf.squeeze(images, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        max_pixel_value = self.value_range[1]
        images = tf.cast(images, dtype=self.compute_dtype)
        images = (images - self.mean * max_pixel_value) / (
            self.std * max_pixel_value
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
                "mean": self.mean_input,
                "std": self.std_input,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
