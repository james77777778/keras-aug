import typing

import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomInvert(VisionRandomLayer):
    """Inverts the colors of the given images.

    The equation of the inversion: `y = value_range[1] - x`.

    Args:
        value_range: The range of values the incoming images will have. This is
            typically either `[0, 1]` or `[0, 255]`.
        p: A float specifying the probability. Defaults to `0.5`.
    """

    def __init__(
        self, value_range: typing.Sequence[float], p: float = 0.5, **kwargs
    ):
        super().__init__(**kwargs)
        self.value_range = standardize_value_range(value_range)
        self.p = float(p)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        ops = self.backend
        random_generator = self.random_generator
        p = ops.random.uniform([batch_size], seed=random_generator)
        return p

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        p = transformations

        def invert(images):
            images = ops.cast(images, self.compute_dtype)
            images = ops.numpy.subtract(self.value_range[1], images)
            return images

        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(
            prob, invert(images), ops.cast(images, self.compute_dtype)
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
        config.update({"value_range": self.value_range, "p": self.p})
        return config
