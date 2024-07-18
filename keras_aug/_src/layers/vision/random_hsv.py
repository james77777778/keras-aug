import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_parameter


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomHSV(VisionRandomLayer):
    """Randomly change the hue, saturation and value.

    The input images must be 3 channels.

    References:
    - [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

    Args:
        hue: How much to jitter hue. The gain will be chosen
            uniformly from `[1 - hue, 1 + hue]` or
            `[min, max]` if given the range of hue. Set to `None` to
            deactivate hue jittering. Defaults to `0.015`.
        saturation: How much to jitter saturation. The gain will be chosen
            uniformly from `[1 - saturation, 1 + saturation]` or
            `[min, max]` if given the range of saturation. Set to `None` to
            deactivate saturation jittering. Defaults to `0.7`.
        value: How much to jitter value. The gain will be chosen
            uniformly from `[1 - value, 1 + value]` or
            `[min, max]` if given the range of value. Set to `None` to
            deactivate value jittering. Defaults to `0.4`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        hue: typing.Union[None, float, typing.Sequence[float]] = 0.015,
        saturation: typing.Union[None, float, typing.Sequence[float]] = 0.7,
        value: typing.Union[None, float, typing.Sequence[float]] = 0.4,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hue = standardize_parameter(
            hue, "hue", center=1.0, bound=(0, float("inf"))
        )
        self.saturation = standardize_parameter(
            saturation, "saturation", center=1.0, bound=(0, float("inf"))
        )
        self.value = standardize_parameter(
            value, "value", center=1.0, bound=(0, float("inf"))
        )
        self.data_format = data_format or keras.config.image_data_format()

        if self.hue == (1.0, 1.0):
            self.hue = None
        if self.saturation == (1.0, 1.0):
            self.saturation = None
        if self.value == (1.0, 1.0):
            self.value = None

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        h, s, v = None, None, None

        def generate_params(low, high):
            return ops.random.uniform(
                [batch_size], low, high, seed=random_generator
            )

        if self.hue is not None:
            h = generate_params(self.hue[0], self.hue[1])
        if self.saturation is not None:
            s = generate_params(self.saturation[0], self.saturation[1])
        if self.value is not None:
            v = generate_params(self.value[0], self.value[1])

        return dict(
            hue_gain=h,
            saturation_gain=s,
            value_gain=v,
        )

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations=None, **kwargs):
        ops = self.backend

        hue_gain = transformations["hue_gain"]
        saturation_gain = transformations["saturation_gain"]
        value_gain = transformations["value_gain"]
        c_axis = -1 if self.data_format == "channels_last" else -3
        original_dtype = backend.standardize_dtype(images.dtype)
        images = self.image_backend.transform_dtype(
            images, images.dtype, backend.result_type(original_dtype, float)
        )
        images = ops.image.rgb_to_hsv(images, data_format=self.data_format)
        hue, saturation, value = ops.numpy.split(images, 3, axis=c_axis)
        if hue_gain is not None:
            hue_gain = ops.numpy.expand_dims(hue_gain, axis=(1, 2, 3))
            hue = ops.numpy.mod(ops.numpy.multiply(hue, hue_gain), 1.0)
        if saturation_gain is not None:
            saturation_gain = ops.numpy.expand_dims(
                saturation_gain, axis=(1, 2, 3)
            )
            saturation = ops.numpy.multiply(saturation, saturation_gain)
        if value_gain is not None:
            value_gain = ops.numpy.expand_dims(value_gain, axis=(1, 2, 3))
            value = ops.numpy.multiply(value, value_gain)
        images = ops.numpy.concatenate([hue, saturation, value], axis=c_axis)
        images = ops.numpy.clip(images, 0.0, 1.0)
        images = ops.image.hsv_to_rgb(images, data_format=self.data_format)
        images = self.image_backend.transform_dtype(
            images, images.dtype, original_dtype
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
            {
                "hue": self.hue,
                "saturation": self.saturation,
                "value": self.value,
            }
        )
        return config
