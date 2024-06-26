import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class Rescale(VisionRandomLayer):
    """Rescales the values of the images to a new range

    The rescaling equation: `y = x * scale + offset`.

    Args:
        scale: The scale to apply to the images.
        offset: The offset to apply to the images. Defaults to `0.0`
    """

    def __init__(self, scale: float, offset: float = 0.0, **kwargs):
        super().__init__(has_generator=False, **kwargs)
        self.scale = float(scale)
        self.offset = float(offset)

        if not backend.is_float_dtype(self.compute_dtype):
            dtype = self.dtype_policy
            raise ValueError(
                f"The `dtype` of '{self.__class__.__name__}' must be float. "
                f"Received: dtype={dtype}"
            )

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        original_dtype = backend.standardize_dtype(images.dtype)
        images = self.image_backend.transform_dtype(
            images, backend.result_type(images.dtype, float)
        )
        scale = ops.convert_to_tensor(self.scale, images.dtype)
        offset = ops.convert_to_tensor(self.offset, images.dtype)
        images = ops.numpy.add(ops.numpy.multiply(images, scale), offset)
        images = self.image_backend.transform_dtype(images, original_dtype)
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
        config.update({"scale": self.scale, "offset": self.offset})
        return config
