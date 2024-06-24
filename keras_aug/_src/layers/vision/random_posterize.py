import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomPosterize(VisionRandomLayer):
    """Posterize the input images with a given probability.

    Posterization reduces the number of bits for each color channel.

    Args:
        bits: The number of bits to keep for each channel (0-8).
        p: A float specifying the probability. Defaults to `0.5`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(self, bits: int, p: float = 0.5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.bits = int(bits)
        self.p = float(p)
        self.data_format = data_format or keras.config.image_data_format()

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
            prob, self.image_backend.posterize(images, self.bits), images
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
        config.update({"bits": self.bits, "p": self.p})
        return config
