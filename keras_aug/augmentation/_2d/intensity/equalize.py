import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class Equalize(VectorizedBaseRandomLayer):
    """Performs histogram equalization on a channel-wise basis.

    Args:
        value_range ((int|float, int|float)): The range of values the incoming
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
        images = tf.transpose(images, [0, 3, 1, 2])
        images = tf.vectorized_map(
            self.equalize_single_image,
            images,
        )
        images = tf.transpose(images, [0, 2, 3, 1])
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
        # Compute the histogram of the image channel.
        scaled_image = (image / 255) * (self.bins - 1)
        scaled_image = tf.cast(scaled_image, dtype=tf.int32)
        scaled_image = tf.clip_by_value(scaled_image, 0, self.bins - 1)
        hist = tf.math.bincount(
            tf.reshape(scaled_image, shape=(tf.shape(scaled_image)[0], -1)),
            minlength=self.bins,
            axis=-1,
        )
        # For the purposes of computing the step, filter out the non-zeros.
        # Zeroes are replaced by a big number while calculating min to keep
        # shape constant across input sizes
        big_number = 1410065408
        hist_without_zeroes = tf.where(
            tf.equal(hist, 0),
            big_number,
            hist,
        )
        step = (
            tf.reduce_sum(hist, axis=-1)
            - tf.reduce_min(hist_without_zeroes, axis=-1)
        ) // (self.bins - 1)

        def build_mapping(histogram, step):
            step = step[:, tf.newaxis]
            # avoid division by zero
            step = tf.where(step == 0, 1, step)
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lookup_table = (tf.cumsum(histogram, axis=-1) + (step // 2)) // step
            # Shift lookup_table, prepending with 0.
            lookup_table = tf.concat(
                [
                    tf.zeros(
                        shape=(tf.shape(lookup_table)[0], 1), dtype=tf.int32
                    ),
                    lookup_table[:, :-1],
                ],
                axis=1,
            )
            # Clip the counts to be in range. This is done
            # in the C code for image.point.
            return tf.clip_by_value(lookup_table, 0, 255)

        # If step is zero, return the original image. Otherwise, build
        # lookup table from the full histogram and step and then index from it.
        image = tf.cast(image, dtype=tf.int32)
        image = tf.where(
            step[:, tf.newaxis, tf.newaxis] == 0,
            image,
            tf.gather(build_mapping(hist, step), image, batch_dims=1),
        )
        return image

    def get_config(self):
        config = super().get_config()
        config.update({"value_range": self.value_range, "bins": self.bins})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
