import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_bbox_format
from keras_aug._src.utils.argument_validation import standardize_data_format
from keras_aug._src.utils.argument_validation import standardize_padding_mode
from keras_aug._src.utils.argument_validation import standardize_size


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class Mosaic(VisionRandomLayer):
    """Apply Mosaic augmentation to the provided batch of images and labels.

    The Mosaic data augmentation first takes 4 images from the batch and makes a
    grid.

    Note that `MixUp` is meant to be used on batches of inputs, not individual
    input. The sample pairing is deterministic and done by matching consecutive
    samples in the batch, so the batch needs to be shuffled.

    Typically, `MixUp` expects the `labels` to be one-hot-encoded format. If
    they are not, with provided `num_classes`, this layer will transform the
    `labels` into one-hot-encoded format. (e.g. `(batch_size, num_classes)`)

    Args:
        size: Desired output size of the mosaic. If `size` is an int instead of
            sequence like `(h, w)`, a square `(size, size)` is made.
        offset: The offset of the mosaic center from the top-left corner of the
            mosaic. Defaults to `(0.25, 0.75)`.
        num_classes: The number of classes in the inputs. Used for one-hot
            encoding. Can be `None` if the labels are already one-hot-encoded.
            Defaults to `None`.
        padding_mode: The mode of the padding. Available values:
            `"constant", "edge", "reflect", "symmetric"`. Defaults to
            `"constant"`.
        padding_value: The padding value. It only affects when
            `padding_mode="constant"`. Defaults to `0`.
        bounding_box_format: The format of the bounding boxes. If specified,
            the available values are `"xyxy", "xywh", "center_xywh", "rel_xyxy",
            "rel_xywh", "rel_center_xywh"`. Defaults to `None`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """  # noqa: E501

    def __init__(
        self,
        size: typing.Union[typing.Sequence[int], int],
        offset: typing.Sequence[float] = (0.25, 0.75),
        num_classes: typing.Optional[int] = None,
        padding_mode: str = "constant",
        padding_value: float = 0,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.size = standardize_size(size)
        self.offset = tuple(offset)
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.padding_mode = standardize_padding_mode(padding_mode)
        self.padding_value = padding_value
        self.bounding_box_format = standardize_bbox_format(bounding_box_format)
        self.data_format = standardize_data_format(data_format)

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        dtype = backend.result_type(self.compute_dtype, float)
        lam = ops.random.beta(
            [batch_size], self.alpha, self.alpha, seed=random_generator
        )
        lam = ops.cast(lam, dtype)
        return lam

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend

        lam = transformations
        original_dtype = backend.standardize_dtype(images.dtype)
        dtype = backend.result_type(images.dtype, float)
        images = self.image_backend.transform_dtype(images, dtype)
        rolled_images = ops.numpy.roll(images, shift=1, axis=0)
        lam = ops.numpy.expand_dims(lam, axis=[1, 2, 3])
        images = ops.numpy.add(
            ops.numpy.multiply(rolled_images, 1.0 - lam),
            ops.numpy.multiply(images, lam),
        )
        images = self.image_backend.transform_dtype(images, original_dtype)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        ops = self.backend

        lam = transformations
        compute_dtype = backend.result_type(labels.dtype, float)
        labels_ndim = len(ops.shape(labels))
        if labels_ndim == 1:
            if self.num_classes is None:
                raise ValueError(
                    "If `labels` is not one-hot-encoded, you must provide "
                    "`num_classes` in the constructor. "
                    f"Received: num_classes={self.num_classes}"
                )
            labels = ops.nn.one_hot(
                labels, self.num_classes, axis=-1, dtype=compute_dtype
            )
        labels = ops.cast(labels, compute_dtype)
        rolled_labels = ops.numpy.roll(labels, shift=1, axis=0)
        lam = ops.numpy.expand_dims(lam, axis=-1)
        labels = ops.numpy.add(
            ops.numpy.multiply(rolled_labels, 1.0 - lam),
            ops.numpy.multiply(labels, lam),
        )
        return labels

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "num_classes": self.num_classes})
        return config
