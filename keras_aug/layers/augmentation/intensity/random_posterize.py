import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomPosterize(VectorizedBaseRandomLayer):
    """Randomly reduces the number of bits for each color channel.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        factor (int|Sequence[int]|keras_aug.FactorSampler): The number of bits to
            keep for each channel. Must be a value between ``[1, 8]``.
            ``factor=(5, 8)`` means RandomPosterize will randomly keep 5 to 8
            bits for the image.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `AutoAugment <https://arxiv.org/abs/1805.09501>`_
        - `RandAugment <https://arxiv.org/abs/1909.13719>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(self, value_range, factor, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        if isinstance(factor, int):
            if not (0 < factor < 9):
                raise ValueError(
                    "factor value must be between [1, 8]. "
                    f"Received bits: {factor}."
                )
            factor = (factor, 8 + 1)
        elif isinstance(factor, (tuple, list)):
            factor = (factor[0], factor[1] + 1)
        self.factor = augmentation_utils.parse_factor(
            factor, min_value=0, max_value=8 + 1
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # cannot sample from tf.int32 due to self.factor might be
        # NormalFactorSampler
        factors = self.factor(shape=(batch_size, 1))
        factors = tf.clip_by_value(factors, 0, 8)
        return tf.cast(factors, dtype=tf.int32)

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
        )
        images = tf.cast(images, tf.uint8)

        inputs_for_posterize_single_image = {
            augmentation_utils.IMAGES: images,
            "bits": transformations,
        }
        images = tf.vectorized_map(
            self.posterize_single_image, inputs_for_posterize_single_image
        )
        images = tf.cast(images, self.compute_dtype)
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=(0, 255),
            target_range=self.value_range,
            dtype=self.compute_dtype,
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

    def posterize_single_image(self, inputs):
        image = inputs.get(augmentation_utils.IMAGES, None)
        shift = 8 - tf.cast(inputs.get("bits", None), dtype=tf.uint8)
        return tf.bitwise.left_shift(
            tf.bitwise.right_shift(image, shift), shift
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "factor": self.factor,
                "seed": self.seed,
            }
        )
        return config
