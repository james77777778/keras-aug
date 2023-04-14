import tensorflow as tf
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomGamma(VectorizedBaseImageAugmentationLayer):
    """Randomly adjusts gamma of the input images.

    This layer will randomly increase/reduce the gamma for the input images.
    Gamma is adjusted independently of each image.

    Args:
        value_range: A list or tuple of 2 floats for the lower and upper limit
            of the values of the input data. The gamma adjustment will be
            scaled to this range, and the output values will be clipped to this
            range.
        factor: A positive float represented as fraction of value, or a tuple of
            size 2 representing lower and upper bound. When represented as a
            single float, lower = upper. The gamma factor will be randomly
            picked between `[1.0 - lower, 1.0 + upper]`. For any pixel x in the
            image, the output will be `x ** factor`.
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
        if isinstance(factor, (tuple, list)):
            min = factor[0]
            max = factor[1]
        else:
            min = 1.0 - factor
            max = 1.0 + factor
        self.factor_input = factor

        self.factor = preprocessing_utils.parse_factor(
            (min, max), min_value=0, max_value=None, seed=seed
        )
        self.value_range = value_range
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        factors = self.factor(shape=(batch_size, 1), dtype=tf.float32)
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
            images,
            original_range=self.value_range,
            target_range=(0.0, 1.0),
            dtype=tf.float32,
        )

        factors = transformations
        images = tf.pow(images, factors[:, :, tf.newaxis, tf.newaxis])

        images = preprocessing_utils.transform_value_range(
            images,
            original_range=(0.0, 1.0),
            target_range=self.value_range,
            dtype=tf.float32,
        )
        return tf.cast(images, dtype=self.compute_dtype)

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
                "factor": self.factor_input,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
