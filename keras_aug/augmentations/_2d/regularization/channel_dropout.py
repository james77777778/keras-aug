import tensorflow as tf
from keras_cv.layers.preprocessing.vectorized_base_image_augmentation_layer import (  # noqa: E501
    VectorizedBaseImageAugmentationLayer,
)
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_aug")
class ChannelDropout(VectorizedBaseImageAugmentationLayer):
    """Randomly drop channels of the input images.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        channel_drop_range: A tuple or list of int represents the range from
            which we choose the number of channels to drop, defaults to (0, 2).
        fill_value: The pixel value for the dropped channel, defauls to 0.
        seed: Integer. Used to create a random seed.
    """

    def __init__(
        self, channel_drop_range=[0, 2], fill_value=0, seed=None, **kwargs
    ):
        super().__init__(seed=seed, **kwargs)
        self.channel_drop_range = sorted(channel_drop_range)
        self.fill_value = fill_value
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        channels_to_drop = self._random_generator.random_uniform(
            (batch_size, 1),
            minval=self.channel_drop_range[0],
            maxval=self.channel_drop_range[1] + 1,
            dtype=tf.int32,
        )
        return channels_to_drop

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        channel = tf.shape(images)[-1]
        indices = transformations

        drop_matrix = tf.one_hot(
            indices, depth=channel, on_value=0.0, off_value=1.0, axis=-1
        )
        drop_matrix = tf.expand_dims(drop_matrix, axis=1)
        fill_value_matrix = (
            tf.one_hot(
                indices, depth=channel, on_value=1.0, off_value=0.0, axis=-1
            )
            * self.fill_value
        )
        fill_value_matrix = tf.expand_dims(fill_value_matrix, axis=1)

        images = images * drop_matrix + fill_value_matrix
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
        config = {
            "channel_drop_range": self.channel_drop_range,
            "fill_value": self.fill_value,
            "seed": self.seed,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
