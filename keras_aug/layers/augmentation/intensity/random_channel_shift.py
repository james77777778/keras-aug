import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomChannelShift(VectorizedBaseRandomLayer):
    """Randomly shift values for each channel of the input images.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        factor (float|Sequence[float]|keras_aug.FactorSampler): The range of the
            channel shift factor. When represented as a single float,
            the factor will be picked between ``[0.0 - lower, 0.0 + upper]``.
            ``0.0`` gives the original image.
        channels (int, optional): The number of channels to shift. Defaults to
            ``3`` corresponds to RGB shift.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, value_range, factor, channels=3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.factor = augmentation_utils.parse_factor(
            factor,
            min_value=-1.0,
            max_value=1.0,
            center_value=0.0,
        )
        self.channels = channels
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        shifts = self.factor(
            shape=(batch_size, self.channels), dtype=self.compute_dtype
        )
        return shifts

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        transformations = tf.expand_dims(transformation, axis=0)
        images = self.augment_images(
            images=images, transformations=transformations, **kwargs
        )
        return tf.squeeze(images, axis=0)

    def augment_images(self, images, transformations=None, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, (0, 1), dtype=self.compute_dtype
        )
        shifts = transformations[:, tf.newaxis, tf.newaxis, :]
        images = images + shifts
        images = tf.clip_by_value(images, 0.0, 1.0)
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
                "factor": self.factor,
                "channels": self.channels,
                "seed": self.seed,
            }
        )
        return config
