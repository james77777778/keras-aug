import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomSolarize(VectorizedBaseRandomLayer):
    """Randomly applies ``(max_value - pixel + min_value)`` for each pixel in
    the input images.

    When created without ``threshold_factor`` parameter, the layer performs
    solarization to all values. When created with specified ``threshold_factor``
    the layer only augments pixels that are above the ``threshold_factor``
    value.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        threshold_factor (float|Sequence[float]|keras_aug.FactorSampler):
            The range of the threshold factor. Only the pixel values above the
            threshold will be solarized. When represented as a single float,
            the factor will be picked between ``[0, upper]``. ``255``
            means no thresholding.
        addition_factor (float|Sequence[float]|keras_aug.FactorSampler, optional):
            The range of the addition factor that is added to each pixel before
            solarization and thresholding. When represented as a single float,
            the factor will be picked between ``[0, upper]``. ``0`` means no
            addition. Defaults to ``0``.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `AutoAugment <https://arxiv.org/abs/1805.09501>`_
        - `RandAugment <https://arxiv.org/abs/1909.13719>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
        threshold_factor,
        addition_factor=0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        if isinstance(threshold_factor, (int, float)):
            threshold_factor = (0, threshold_factor)
        self.threshold_factor = augmentation_utils.parse_factor(
            threshold_factor,
            max_value=255,
            seed=seed,
            param_name="threshold_factor",
        )
        if isinstance(addition_factor, (int, float)):
            addition_factor = (0, addition_factor)
        self.addition_factor = augmentation_utils.parse_factor(
            addition_factor,
            max_value=255,
            seed=seed,
            param_name="addition_factor",
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        return {
            "additions": self.addition_factor(
                shape=(batch_size, 1, 1, 1), dtype=self.compute_dtype
            ),
            "thresholds": self.threshold_factor(
                shape=(batch_size, 1, 1, 1), dtype=self.compute_dtype
            ),
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        thresholds = transformations["thresholds"]
        additions = transformations["additions"]
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        images = images + additions
        images = tf.clip_by_value(images, 0, 255)
        images = tf.where(images < thresholds, images, 255 - images)
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

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "threshold_factor": self.threshold_factor,
                "addition_factor": self.addition_factor,
                "seed": self.seed,
            }
        )
        return config
