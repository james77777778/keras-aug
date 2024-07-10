import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomGrayscale(VisionRandomLayer):
    """Randomly convert the images to grayscale.

    The input images must be 3 channels.

    Args:
        p: A float specifying the probability. Defaults to `0.5`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(self, p: float = 0.5, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.p = float(p)
        self.data_format = data_format or keras.config.image_data_format()

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        p = ops.random.uniform([batch_size], seed=random_generator)
        return p

    def compute_output_shape(self, input_shape):
        return input_shape

    def augment_images(self, images, transformations=None, **kwargs):
        ops = self.backend
        p = transformations
        prob = ops.numpy.expand_dims(p < self.p, axis=[1, 2, 3])
        images = ops.numpy.where(
            prob,
            self.image_backend.rgb_to_grayscale(
                images, data_format=self.data_format
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
        config.update({"p": self.p})
        return config
