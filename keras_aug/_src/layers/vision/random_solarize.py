import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomSolarize(VisionRandomLayer):
    """Solarize the input images with a given probability

    Solarization inverts all pixel values above a threshold.

    Args:
        threshold: All pixels equal or above this value are inverted.
        p: A float specifying the probability. Defaults to `0.5`.
    """

    def __init__(self, threshold: float, p: float = 0.5, **kwargs):
        super().__init__(**kwargs)
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

        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(
            prob,
            self.image_backend.solarize(images, self.threshold),
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
        config.update({"threshold": self.threshold, "p": self.p})
        return config
