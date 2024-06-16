import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomPosterize(VisionRandomLayer):
    """Posterize the input images with a given probability.

    Posterization reduces the number of bits for each color channel.

    Args:
        value_range: The range of values the incoming images will have. This is
            typically either `[0, 1]` or `[0, 255]`.
        bits: The number of bits to keep for each channel (0-8).
        p: A float specifying the probability. Defaults to `0.5`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        value_range: typing.Sequence[float],
        bits: int,
        p: float = 0.5,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = standardize_value_range(value_range)
        self.bits = int(bits)
        self.p = float(p)
        self.data_format = data_format or keras.config.image_data_format()

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        p = ops.random.uniform([batch_size], seed=random_generator)
        return p

    def augment_images(self, images, transformations=None, **kwargs):
        ops = self.backend
        p = transformations

        images = ops.convert_to_tensor(images)
        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        if backend.is_float_dtype(images.dtype):
            images = self.transform_value_range(
                images, self.value_range, (0, 1), self.compute_dtype
            )

        images = ops.cast(images, self.compute_dtype)
        images = ops.numpy.where(
            prob, self.image_backend.posterize(images, self.bits), images
        )
        if backend.is_float_dtype(images.dtype):
            images = self.transform_value_range(
                images, (0, 1), self.value_range, self.compute_dtype
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
            {"value_range": self.value_range, "bits": self.bits, "p": self.p}
        )
        return config
