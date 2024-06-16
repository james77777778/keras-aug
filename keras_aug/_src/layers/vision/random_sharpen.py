import typing

import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomSharpen(VisionRandomLayer):
    """Adjust the sharpness of the input images with a given probability.

    Args:
        value_range: The range of values the incoming images will have. This is
            typically either `[0, 1]` or `[0, 255]`.
        sharpness_factor: How much to adjust the sharpness. Can be any
            non-negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
        p: A float specifying the probability. Defaults to `0.5`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        value_range: typing.Sequence[float],
        sharpness_factor: float,
        p: float = 0.5,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value_range = standardize_value_range(value_range)
        self.sharpness_factor = float(sharpness_factor)
        self.p = float(p)
        self.data_format = data_format or keras.config.image_data_format()

        if self.sharpness_factor < 0:
            raise ValueError(
                "`sharpness_factor` should be a non-negative number. "
                f"Received: sharpness_factor={sharpness_factor}"
            )

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
        images = ops.cast(images, self.compute_dtype)
        images = ops.numpy.where(
            prob,
            self.image_backend.sharpen(
                images,
                self.sharpness_factor,
                self.value_range,
                self.data_format,
            ),
            images,
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
                "value_range": self.value_range,
                "sharpness_factor": self.sharpness_factor,
                "p": self.p,
            }
        )
        return config
