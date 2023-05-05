import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class Rescale(VectorizedBaseRandomLayer):
    """Rescales the inputs to a new range.

    Rescale rescales every value of the inputs (often the images) by the
    equation: ``y = x * scale + offset``.

    Args:
        scale (int|float): The scale to apply to the inputs.
        offset (int|float, optional): The offset to apply to the inputs.
            Defaults to ``0.0``

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, scale, offset=0.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = tf.cast(scale, self.compute_dtype)
        self.offset = tf.cast(offset, self.compute_dtype)

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        images = self.augment_images(
            images=images, transformations=transformation, **kwargs
        )
        return tf.squeeze(images, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = tf.cast(images, dtype=self.compute_dtype)
        return images * self.scale + self.offset

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
        config.update({"scale": self.scale, "offset": self.offset})
        return config
