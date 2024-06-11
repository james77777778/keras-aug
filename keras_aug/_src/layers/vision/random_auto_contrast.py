import typing

import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomAutoContrast(VisionRandomLayer):
    """Autocontrast the images randomly with a given probability.

    Auto contrast stretches the values of an image across the entire available
    `value_range`. This makes differences between pixels more obvious. An
    example of this is if an image only has values `[0, 1]` out of the range
    `[0, 255]`, auto contrast will change the `1` values to be `255`.

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

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        p = ops.random.uniform([batch_size], seed=random_generator)
        return p

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        p = transformations

        def auto_contrast(images):
            images = ops.cast(images, "float32")
            lows = ops.numpy.min(images, axis=(1, 2), keepdims=True)
            highs = ops.numpy.max(images, axis=(1, 2), keepdims=True)
            scales = (self.value_range[1] - self.value_range[0]) / (
                highs - lows
            )
            eq_idxs = ops.numpy.isinf(scales)
            lows = ops.numpy.where(eq_idxs, 0.0, lows)
            scales = ops.numpy.where(eq_idxs, 1.0, scales)
            images = ops.numpy.clip(
                (images - lows) * scales,
                self.value_range[0],
                self.value_range[1],
            )
            images = ops.cast(images, self.compute_dtype)
            return images

        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(
            prob, auto_contrast(images), ops.cast(images, self.compute_dtype)
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
