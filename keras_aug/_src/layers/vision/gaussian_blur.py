import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_parameter


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class GaussianBlur(VisionRandomLayer):
    """Blurs the input images with randomly chosen Gaussian blur kernel.

    The convolution will be using 'reflect' padding corresponding to the
    kernel size, to maintain the input shape.

    Note that due to implementation limitations, a single sampled `sigma` will
    be applied to the entire batch of images.

    Args:
        kernel_size: An int or a sequence of ints specifying the size of the
            Gaussian kernel in x and y directions. The values should be odd and
            positive numbers.
        sigma: A float or a sequence of floats specifying standard deviation to
            be used for the Gaussian kernel. If float, sigma is fixed. If a
            sequence of floats, sigma is sampled uniformly in the given range.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(
        self,
        kernel_size: typing.Union[int, typing.Sequence[int]],
        sigma: typing.Union[float, typing.Sequence[float]] = (0.1, 2.0),
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(sigma, (int, float)):
            sigma = (float(sigma), float(sigma))
        self.kernel_size = tuple(kernel_size)
        self.sigma = tuple(sigma)
        self.data_format = data_format or keras.config.image_data_format()

        if len(kernel_size) != 2:
            raise ValueError(
                "The length of `kernel_size` should be 2. "
                f"Received: kernel_size={kernel_size}"
            )
        for ks in kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError(
                    "The values of `kernel_size` should be odd and positive."
                )
        standardize_parameter(
            self.sigma,
            "sigma",
            bound=(0.0, float("inf")),
            allow_none=False,
            allow_single_number=False,
        )

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        dtype = backend.result_type(images.dtype, float)
        sigma = ops.random.uniform(
            [1],
            self.sigma[0],
            self.sigma[1],
            dtype=dtype,
            seed=random_generator,
        )
        return sigma[0]

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations=None, **kwargs):
        sigma = transformations
        images = self.image_backend.guassian_blur(
            images,
            self.kernel_size,
            [sigma, sigma],
            data_format=self.data_format,
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
        config.update({"kernel_size": self.kernel_size, "sigma": self.sigma})
        return config
