import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_data_format


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class MixUp(VisionRandomLayer):
    """Apply MixUp to the provided batch of images and labels.

    Note that `MixUp` is meant to be used on batches of inputs, not individual
    input. The sample pairing is deterministic and done by matching consecutive
    samples in the batch, so the batch needs to be shuffled.

    Typically, `MixUp` expects the `labels` to be one-hot-encoded format. If
    they are not, with provided `num_classes`, this layer will transform the
    `labels` into one-hot-encoded format. (e.g. `(batch_size, num_classes)`)

    References:
    - [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

    Args:
        alpha: The hyperparameter of the beta distribution used for cutmix.
            Defaults to `1.0`.
        num_classes: The number of classes in the inputs. Used for one-hot
            encoding. Can be `None` if the labels are already one-hot-encoded.
            Defaults to `None`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """  # noqa: E501

    def __init__(
        self,
        alpha: float = 1.0,
        num_classes: typing.Optional[int] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = float(alpha)
        self.num_classes = int(num_classes) if num_classes is not None else None
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
