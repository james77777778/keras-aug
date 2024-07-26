import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_data_format


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomAutoContrast(VisionRandomLayer):
    """Autocontrast the images randomly with a given probability.

    Auto contrast stretches the values of an image across the entire available
    value range. This makes differences between pixels more obvious.

    Args:
        p: A float specifying the probability. Defaults to `0.5`.
    """

    def __init__(self, p: float = 0.5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)
        self.data_format = standardize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        p = ops.random.uniform([batch_size], seed=random_generator)
        return p

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        p = transformations

        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(
            prob,
            self.image_backend.auto_contrast(images, self.data_format),
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
        config.update({"p": self.p})
        return config
