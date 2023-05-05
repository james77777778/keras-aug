import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class ChannelShuffle(VectorizedBaseRandomLayer):
    """Shuffles channels of the input images.

    Args:
        groups (int, optional): The number of the groups to divide the input
            channels. Defaults to ``3``.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, groups=3, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.groups = groups
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # get batched shuffled indices
        # for example: batch_size=2; self.group=5
        # indices = [
        #     [0, 2, 3, 4, 1],
        #     [4, 1, 0, 2, 3]
        # ]
        indices = self._random_generator.random_uniform(
            (batch_size, self.groups)
        )
        indices = tf.argsort(indices, axis=-1)
        return indices

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        batch_size = tf.shape(images)[0]
        height, width = tf.shape(images)[1], tf.shape(images)[2]
        num_channels = images.shape[3]
        indices = transformations

        # append batch indexes next to shuffled indices
        batch_indexs = tf.repeat(tf.range(batch_size), self.groups)
        batch_indexs = tf.reshape(batch_indexs, (batch_size, self.groups))
        indices = tf.stack([batch_indexs, indices], axis=-1)

        if not num_channels % self.groups == 0:
            raise ValueError(
                "The number of input channels should be "
                "divisible by the number of groups."
                f"Received: channels={num_channels}, groups={self.groups}"
            )
        images = tf.reshape(
            images, [batch_size, height, width, self.groups, -1]
        )
        images = tf.transpose(images, perm=[0, 3, 1, 2, 4])
        images = tf.gather_nd(images, indices=indices)
        images = tf.transpose(images, perm=[0, 2, 3, 4, 1])
        images = tf.reshape(images, [batch_size, height, width, num_channels])
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
            {
                "groups": self.groups,
                "seed": self.seed,
            }
        )
        return config
