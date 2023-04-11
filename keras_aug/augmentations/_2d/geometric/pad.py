import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from tensorflow import keras

H_AXIS = -3
W_AXIS = -2


@keras.utils.register_keras_serializable(package="keras_aug")
class Pad(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly crops images.
    This layer will randomly choose a location to crop images down to a target
    size.
    If an input image is smaller than the target size, the input will be
    resized and cropped to return the largest possible window in the image that
    matches the target aspect ratio.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of interger or floating point dtype.
    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., target_height, target_width, channels)`.
    Args:
        height: Integer, the height of the output shape.
        width: Integer, the width of the output shape.
        seed: Integer. Used to create a random seed.
    """

    def __init__(self, seed=None, bounding_box_format=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.seed = seed
        self.bounding_box_format = bounding_box_format

    def get_random_transformation_batch(self, batch_size, **kwargs):
        pass
        return tf.zeros((batch_size,))

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`Pad()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`Pad(..., bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        return bounding_boxes

    def _get_image_shape(self, images):
        if isinstance(images, tf.RaggedTensor):
            heights = tf.reshape(images.row_lengths(), (-1, 1))
            widths = tf.reshape(
                tf.reduce_max(images.row_lengths(axis=2), 1), (-1, 1)
            )
        else:
            batch_size = tf.shape(images)[0]
            heights = tf.repeat(tf.shape(images)[H_AXIS], repeats=[batch_size])
            heights = tf.reshape(heights, shape=(-1, 1))
            widths = tf.repeat(tf.shape(images)[W_AXIS], repeats=[batch_size])
            widths = tf.reshape(widths, shape=(-1, 1))
        return tf.cast(heights, dtype=tf.int32), tf.cast(widths, dtype=tf.int32)

    def get_config(self):
        config = {
            "height": self.height,
            "width": self.width,
            "seed": self.seed,
            "bounding_box_format": self.bounding_box_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
