import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class GaussianNoise(VisionRandomLayer):
    """Add gaussian noise to the input images.

    Args:
        mean: Mean of the sampled normal distribution. Defaults to `0.0`.
        sigma: Standard deviation of the sampled normal distribution. Defaults
            to `0.1`.
        clip: Whether to clip the values in `[0, 1]`. Defaults to `True`.
    """

    def __init__(
        self, mean: float = 0.0, sigma: float = 0.1, clip: bool = True, **kwargs
    ):
        super().__init__(**kwargs)
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.clip = bool(clip)

        if not backend.is_float_dtype(self.compute_dtype):
            dtype = self.dtype_policy
            raise ValueError(
                f"The `dtype` of '{self.__class__.__name__}' must be float. "
                f"Received: dtype={dtype}"
            )

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        dtype = backend.result_type(images.dtype, float)
        noise = (
            self.mean
            + ops.random.normal(
                ops.shape(images), dtype=dtype, seed=random_generator
            )
            * self.sigma
        )
        return noise

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        original_dtype = backend.standardize_dtype(images.dtype)
        noise = transformations

        images = self.image_backend.transform_dtype(
            images, images.dtype, backend.result_type(images.dtype, float)
        )
        images = ops.numpy.add(images, noise)
        if self.clip:
            images = ops.numpy.clip(images, 0, 1)
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
            {"mean": self.mean, "sigma": self.sigma, "clip": self.clip}
        )
        return config
