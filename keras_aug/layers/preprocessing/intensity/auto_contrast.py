import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class AutoContrast(VectorizedBaseRandomLayer):
    """Performs the AutoContrast operation on the input images.

    Auto contrast stretches the values of an image across the entire available
    ``value_range``. This makes differences between pixels more obvious. An
    example of this is if an image only has values ``[0, 1]`` out of the range
    ``[0, 255]``, auto contrast will change the ``1`` values to be ``255``.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, value_range, **kwargs):
        super().__init__(**kwargs)
        self.value_range = value_range

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        images = self.augment_images(
            images=images, transformations=transformation, **kwargs
        )
        return tf.squeeze(images, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        lows = tf.reduce_min(images, axis=(1, 2), keepdims=True)
        highs = tf.reduce_max(images, axis=(1, 2), keepdims=True)
        scales = 255.0 / (highs - lows)
        eq_idxs = tf.math.is_inf(scales)
        lows = tf.where(eq_idxs, 0.0, lows)
        scales = tf.where(eq_idxs, 1.0, scales)
        images = tf.clip_by_value((images - lows) * scales, 0, 255)
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
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
        config.update({"value_range": self.value_range})
        return config
