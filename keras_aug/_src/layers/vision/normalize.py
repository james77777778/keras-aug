import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class Normalize(VisionRandomLayer):
    """Normalize the images with mean and standard deviation.

    This layer will normalize each channel of the images:
    `y[c] = (x[c] - mean[c]) / std[c]`.

    Args:
        mean: Sequence of means for each channel. Defaults to
            `(0.485, 0.456, 0.406)` which is the mean values from ImageNet.
        std: Sequence of standard deviations for each channel. Defaults to
            `(0.229, 0.224, 0.225)` which is the std values from ImageNet.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        mean: typing.Sequence[float] = (0.485, 0.456, 0.406),
        std: typing.Sequence[float] = (0.229, 0.224, 0.225),
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(has_generator=False, **kwargs)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.data_format = data_format or backend.image_data_format()

        if not backend.is_float_dtype(self.compute_dtype):
            dtype = self.dtype_policy
            raise ValueError(
                "The `dtype` of Normalize must be float. "
                f"Received: dtype={dtype}"
            )

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        original_dtype = backend.standardize_dtype(images.dtype)
        if self.data_format == "channels_last":
            mean = ops.numpy.expand_dims(self.mean, axis=[0, 1, 2])
            std = ops.numpy.expand_dims(self.std, axis=[0, 1, 2])
        else:
            mean = ops.numpy.expand_dims(self.mean, axis=[0, 2, 3])
            std = ops.numpy.expand_dims(self.std, axis=[0, 2, 3])
        images = ops.numpy.subtract(images, mean)
        images = ops.numpy.divide(images, std)
        return ops.cast(images, original_dtype)

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
        config.update({"mean": self.mean, "std": self.std})
        return config
