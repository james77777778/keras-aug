import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class Grayscale(VectorizedBaseRandomLayer):
    """Grayscale transforms RGB images to grayscale images.

    Args:
        output_channels (int, optional): The number of the color channels of
            the outputs. Defaults to ``3``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, output_channels=3, **kwargs):
        super().__init__(**kwargs)
        if output_channels not in (1, 3):
            raise ValueError(
                "Received invalid argument output_channels. "
                f"output_channels must be in 1 or 3. Got {output_channels}"
            )
        self.output_channels = output_channels

    def compute_ragged_image_signature(self, images):
        ragged_spec = tf.RaggedTensorSpec(
            shape=images.shape[1:3] + (self.output_channels,),
            ragged_rank=1,
            dtype=self.compute_dtype,
        )
        return ragged_spec

    def augment_ragged_image(self, image, transformation, **kwargs):
        return self.augment_images(
            image, transformations=transformation, **kwargs
        )

    def augment_images(self, images, transformations=None, **kwargs):
        grayscales = augmentation_utils.rgb_to_grayscale(images)
        if self.output_channels == 1:
            return grayscales
        elif self.output_channels == 3:
            return tf.image.grayscale_to_rgb(grayscales)
        else:
            raise ValueError("Unsupported value for `output_channels`.")

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
        config.update({"output_channels": self.output_channels})
        return config
