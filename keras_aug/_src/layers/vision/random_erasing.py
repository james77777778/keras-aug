import math
import numbers
import typing
import warnings
from collections.abc import Sequence

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_data_format
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomErasing(VisionRandomLayer):
    """Randomly select a rectangle region in the input images erase its pixels.

    If image size is smaller than output size along any edge, image is padded
    with `padding_value` and then cropped.

    References:
    - [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896)

    Args:
        p: A float specifying the probability. Defaults to `0.5`.
        scale: A tuple of floats specifying the lower and upper bounds for the
            range of proportion of erased area against input images. Defaults to
            `(0.02, 0.33)`.
        ratio: A tuple of floats specifying the lower and upper bounds for the
            range of aspect ratio of erased area. Defaults to `(0.3, 3.3)`.
        value: The erasing values. If a single float, it is used to erase all
            pixels. If a tuple of floats, it is used to erase each channels
            respectively. If a str of `"random", erasing each pixel with random
            values. `Defaults to `0`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: typing.Sequence[float] = (0.02, 0.33),
        ratio: typing.Sequence[float] = (0.3, 3.3),
        value: typing.Union[float, typing.Sequence[float], str] = 0.0,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Check
        if isinstance(value, str) and value != "random":
            raise ValueError(
                "If `value` is a string, it must be `'random'`. "
                f"Received: value={value}"
            )
        if isinstance(value, numbers.Number):
            value = float(value)
        if not isinstance(value, str) and isinstance(value, Sequence):
            value = tuple(value)
        self.p = float(p)
        self.scale = standardize_value_range(scale)
        self.ratio = standardize_value_range(ratio)
        self.value = value
        self.data_format = standardize_data_format(data_format)

        self.log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        if self.data_format == "channels_last":
            self.h_axis, self.w_axis, self.c_axis = -3, -2, -1
        else:
            self.c_axis, self.h_axis, self.w_axis = -3, -2, -1
        if self.value == "random" and backend.is_int_dtype(self.compute_dtype):
            raise ValueError(
                "If `value` is 'random', the compute dtype of this layer "
                "should be float. "
                f"Received: value={value}, compute_dtype={self.compute_dtype}"
            )

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, dict):
            images_shape = input_shape[self.IMAGES]
        else:
            images_shape = input_shape
        channels = images_shape[self.c_axis]
        if isinstance(self.value, tuple) and len(self.value) != channels:
            raise ValueError(
                "If `value` is a sequence, the length of `value` must be the "
                "same as channels of input images. "
                f"Received: value={self.value}, images.shape={images_shape}"
            )
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        images_shape = ops.shape(images)
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        area = ops.cast(height * width, "float32")
        scale = ops.random.uniform(
            shape=[batch_size],
            minval=self.scale[0],
            maxval=self.scale[1],
            seed=random_generator,
        )
        target_area = area * scale
        log_ratio = ops.random.uniform(
            shape=[batch_size],
            minval=self.log_ratio[0],
            maxval=self.log_ratio[1],
            seed=random_generator,
        )
        aspect_ratio = ops.numpy.exp(log_ratio)

        w = ops.cast(
            ops.numpy.round(ops.numpy.sqrt(target_area * aspect_ratio)), "int32"
        )
        h = ops.cast(
            ops.numpy.round(ops.numpy.sqrt(target_area / aspect_ratio)), "int32"
        )
        w = ops.numpy.clip(w, 1, width)
        h = ops.numpy.clip(h, 1, height)
        i = ops.random.uniform(shape=[batch_size], seed=random_generator)
        i = ops.cast(
            ops.numpy.round(ops.numpy.multiply(i, (height - h + 1))), "int32"
        )
        j = ops.random.uniform(shape=[batch_size], seed=random_generator)
        j = ops.cast(
            ops.numpy.round(ops.numpy.multiply(j, (width - w + 1))), "int32"
        )

        # Get values
        if isinstance(self.value, str):
            dtype = backend.result_type(images.dtype, float)
            v = ops.random.normal(
                ops.shape(images), dtype=dtype, seed=random_generator
            )
        elif isinstance(self.value, float):
            dtype = backend.standardize_dtype(images.dtype)
            v = ops.numpy.full(ops.shape(images), self.value, dtype=dtype)
        elif isinstance(self.value, tuple):
            dtype = backend.standardize_dtype(images.dtype)
            v = ops.convert_to_tensor(self.value, dtype=dtype)  # [c]
            if self.data_format == "channels_last":
                v = ops.numpy.expand_dims(v, axis=[0, 1, 2])
                v = ops.numpy.tile(v, [batch_size, height, width, 1])
            else:
                v = ops.numpy.expand_dims(v, axis=[0, -1, -2])
                v = ops.numpy.tile(v, [batch_size, 1, height, width])
        return dict(top=i, left=j, height=h, width=w, value=v)

    def augment_images(self, images, transformations, **kwargs):
        left = transformations["left"]
        top = transformations["top"]
        width = transformations["width"]
        height = transformations["height"]
        right = left + width
        bottom = top + height
        value = transformations["value"]
        images = self.image_backend.fill_rectangles(
            images,
            value,
            (left, top, right, bottom),
            data_format=self.data_format,
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformations,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        warnings.warn(
            f"{self.__class__.__name__} is currently passing through "
            "`bounding_boxes`. This will likely change in the future."
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        warnings.warn(
            f"{self.__class__.__name__} is currently passing through "
            "`segmentation_masks`. This will likely change in the future."
        )
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "p": self.p,
                "scale": self.scale,
                "ratio": self.ratio,
                "value": self.value,
            }
        )
        return config
