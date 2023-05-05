from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class Identity(VectorizedBaseRandomLayer):
    """Applies nothing to the inputs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def augment_ragged_image(self, image, transformation, **kwargs):
        return image

    def augment_images(self, images, transformations, **kwargs):
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
        return super().get_config()
