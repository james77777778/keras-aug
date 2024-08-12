import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class ToDType(VisionRandomLayer):
    """Converts the input to a specific dtype, optionally scaling the values.

    If `scale` is `True`, the value range will changed as follows:
    - `"uint8"`: `[0, 255]`
    - `"int16"`: `[-32768, 32767]`
    - `"int32"`: `[-2147483648, 2147483647]`
    - float: `[0.0, 1.0]`

    Args:
        to_dtype: A string specifying the target dtype.
        scale: Whether to scale the values. Defaults to `False`.
    """

    def __init__(self, to_dtype, scale=False, **kwargs):
        to_dtype = backend.standardize_dtype(to_dtype)
        self.scale = bool(scale)
        if "dtype" in kwargs:
            kwargs.pop("dtype")
        super().__init__(has_generator=False, dtype=to_dtype, **kwargs)
        self.to_dtype = to_dtype
        self.transform_dtype_scale = self.scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations, **kwargs):
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
        config.update({"to_dtype": self.to_dtype, "scale": self.scale})
        return config
