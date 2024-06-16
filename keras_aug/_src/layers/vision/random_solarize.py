import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomSolarize(VisionRandomLayer):
    """Solarize the input images with a given probability

    Solarization inverts all pixel values above a threshold.

    Args:
        value_range: The range of values the incoming images will have. This is
            typically either `[0, 1]` or `[0, 255]`.
        threshold: All pixels equal or above this value are inverted.
        p: A float specifying the probability. Defaults to `0.5`.
    """

    def __init__(
        self,
        value_range: typing.Sequence[float],
        threshold: float,
        p: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = standardize_value_range(value_range)
        self.threshold = float(threshold)
        self.p = float(p)

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

        images = ops.cast(images, self.compute_dtype)
        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(
            prob,
            self.image_backend.solarize(
                images, self.threshold, self.value_range
            ),
            images,
        )
        return images

    def _get_dtype_bits(self, dtype):
        dtype = backend.standardize_dtype(dtype)
        if dtype == "uint8":
            return 8
        elif dtype == "uint16":
            return 15
        elif dtype == "uint32":
            return 32
        elif dtype == "uint64":
            return 64
        elif dtype == "int8":
            return 7
        elif dtype == "int16":
            return 15
        elif dtype == "int32":
            return 31
        elif dtype == "int64":
            return 63
        else:
            raise NotImplementedError

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
                "value_range": self.value_range,
                "threshold": self.threshold,
                "p": self.p,
            }
        )
        return config
