import keras

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomChannelPermutation(VisionRandomLayer):
    """Randomly permute the channels of the input images.

    Args:
        num_channels: The number of channels to permute.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """

    def __init__(self, num_channels: int, data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = int(num_channels)
        self.data_format = data_format or keras.config.image_data_format()

        self.channels_axis = -1 if self.data_format == "channels_last" else -3

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        perm = ops.random.uniform(
            [batch_size, self.num_channels], seed=random_generator
        )
        perm = ops.numpy.argsort(perm, axis=-1)
        return perm

    def compute_output_shape(self, input_shape):
        images_shape, _ = self._get_shape_or_spec(input_shape)
        if images_shape[self.channels_axis] != self.num_channels:
            raise ValueError(
                "`num_channels` must match the channels of the input images. "
                f"Received: images.shape={images_shape}, "
                f"num_channels={self.num_channels}"
            )
        return input_shape

    def augment_images(self, images, transformations=None, **kwargs):
        ops = self.backend
        perm = transformations
        if self.data_format == "channels_last":
            perm = ops.numpy.expand_dims(perm, axis=[1, 2])
        else:
            perm = ops.numpy.expand_dims(perm, axis=[2, 3])
        images = ops.numpy.take_along_axis(
            images, perm, axis=self.channels_axis
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
        config.update({"num_channels": self.num_channels})
        return config
