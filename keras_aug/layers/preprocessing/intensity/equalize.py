import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class Equalize(VectorizedBaseRandomLayer):
    """Performs histogram equalization on a channel-wise basis.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        bins (int, optional): The number of bins to use in histogram
            equalization. Should be in the range ``[0, 256]``. Defaults to
            ``256``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, value_range, bins=256, **kwargs):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.bins = bins

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations=None, **kwargs):
        original_shape = images.shape
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, (0, 255), dtype=self.compute_dtype
        )
        images = tf.cast(images, dtype=tf.int32)
        images = tf.map_fn(
            self.equalize_single_image,
            images,
        )
        images = tf.transpose(images, (0, 2, 3, 1))
        images = tf.cast(images, dtype=self.compute_dtype)
        images = preprocessing_utils.transform_value_range(
            images, (0, 255), self.value_range, dtype=self.compute_dtype
        )
        images.set_shape(original_shape)
        return tf.cast(images, dtype=self.compute_dtype)

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

    def equalize_single_image(self, image):
        return tf.map_fn(
            lambda channel_index: self.equalize_single_channel(
                image, channel_index
            ),
            tf.range(tf.shape(image)[-1]),
        )

    def equalize_single_channel(self, image, channel_index):
        image = image[..., channel_index]
        # Compute the histogram of the image channel.
        histogram = tf.histogram_fixed_width(image, [0, 255], nbins=self.bins)
        # For the purposes of computing the step, filter out the non-zeros.
        # Zeroes are replaced by a big number while calculating min to keep
        # shape constant across input sizes for compatibility with
        # vectorized_map
        big_number = 1410065408
        histogram_without_zeroes = tf.where(
            tf.equal(histogram, 0),
            big_number,
            histogram,
        )
        step = (
            tf.reduce_sum(histogram) - tf.reduce_min(histogram_without_zeroes)
        ) // (self.bins - 1)

        def build_mapping(histogram, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lookup_table = (tf.cumsum(histogram) + (step // 2)) // step
            # Shift lookup_table, prepending with 0.
            lookup_table = tf.concat([[0], lookup_table[:-1]], 0)
            # Clip the counts to be in range. This is done
            # in the C code for image.point.
            return tf.clip_by_value(lookup_table, 0, 255)

        # If step is zero, return the original image. Otherwise, build
        # lookup table from the full histogram and step and then index from it.
        image = tf.cond(
            tf.equal(step, 0),
            lambda: image,
            lambda: tf.gather(build_mapping(histogram, step), image),
        )
        return image

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range, "bins": self.bins})
        return config
