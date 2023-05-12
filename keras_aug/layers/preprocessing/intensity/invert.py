import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class Invert(VectorizedBaseRandomLayer):
    """Inverts the inputs.

    Inverts the pixel value by equation: ``y = max_pixel_value - x``.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
    """

    def __init__(self, value_range, **kwargs):
        super().__init__(**kwargs)
        self.value_range = value_range

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            images=image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations, **kwargs):
        images = tf.cast(images, dtype=self.compute_dtype)
        return self.value_range[1] - images

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
