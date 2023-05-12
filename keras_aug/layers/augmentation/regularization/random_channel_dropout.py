import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomChannelDropout(VectorizedBaseRandomLayer):
    """Randomly drop channels of the input images.

    Args:
        factor (float|Sequence[float]|keras_aug.FactorSampler): The range from
            which we choose the number of channels to drop. Defaults to
            ``(0, 2)``.
        fill_value (int|float, optional): The value to be filled for dropped
            channel. Defaults to ``0``.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `Albumentations <https://github.com/albumentations-team/albumentations>`_
    """  # noqa: E501

    def __init__(self, factor=(0, 2), fill_value=0, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if isinstance(factor, (tuple, list)):
            lower = factor[0]
            upper = factor[1] + 1
        else:
            lower = 0
            upper = factor + 1
        self.factor_input = factor
        self.factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=0, max_value=None, seed=seed
        )
        self.fill_value = fill_value
        self.seed = seed

        self.upper = upper

    def get_random_transformation_batch(self, batch_size, **kwargs):
        channels_to_drop = self.factor(shape=(batch_size, 1), dtype=tf.int32)
        return channels_to_drop

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = tf.cast(images, dtype=self.compute_dtype)
        channel = images.shape[-1]
        indices = transformations

        if self.upper > channel:
            raise ValueError(
                f"The number of the channel: {channel} must <= the upper "
                f"bound: {self.upper}."
            )

        drop_matrix = tf.one_hot(
            indices,
            channel,
            on_value=0,
            off_value=1,
            axis=-1,
            dtype=self.compute_dtype,
        )
        drop_matrix = tf.expand_dims(drop_matrix, axis=1)
        fill_value_matrix = (
            tf.one_hot(
                indices,
                channel,
                on_value=1,
                off_value=0,
                axis=-1,
                dtype=self.compute_dtype,
            )
            * self.fill_value
        )
        fill_value_matrix = tf.expand_dims(fill_value_matrix, axis=1)
        return images * drop_matrix + fill_value_matrix

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
        config = {
            "factor": self.factor_input,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
