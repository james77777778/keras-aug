import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentations._2d.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomGamma(VectorizedBaseRandomLayer):
    """Randomly adjusts gamma of the input images.

    This layer will randomly increase/reduce the gamma for the input images.
    Gamma is adjusted independently of each image.

    Args:
        value_range: A list or tuple of 2 floats for the lower and upper limit
            of the values of the input data. The gamma adjustment will be
            scaled to this range, and the output values will be clipped to this
            range.
        factor: A tuple of two floats, a single float or
            `keras_cv.FactorSampler`. When represented as a single float,
            lower = upper. The factor will be randomly picked between
            `[1.0 - lower, 1.0 + upper]`. 1.0 will give the original
            image. For any pixel x in the image, the output will be
            `x ** factor`.
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        value_range,
        factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.factor = augmentation_utils.parse_factor(
            factor, max_value=None, center_value=1.0, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        factors = self.factor(shape=(batch_size, 1), dtype=self.compute_dtype)
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
            images, self.value_range, (0.0, 1.0), dtype=self.compute_dtype
        )
        factors = transformations
        images = tf.pow(images, factors[:, :, tf.newaxis, tf.newaxis])
        images = preprocessing_utils.transform_value_range(
            images, (0.0, 1.0), self.value_range, dtype=self.compute_dtype
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
                "factor": self.factor,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
