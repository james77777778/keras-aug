import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomSharpness(VectorizedBaseRandomLayer):
    """Randomly performs the sharpness operation on given images.

    The sharpness operation first performs a blur operation, then blends between
    the original image and the blurred image. This operation makes the edges of
    an image less sharp than they were in the original image.

    Args:
        value_range (Sequence[int|float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        factor (float|Sequence[float]|keras_aug.FactorSampler): The range of the
            sharpness factor. When represented as a single float, the factor
            will be picked between ``[1.0 - lower, 1.0 + upper]``. ``1.0`` will
            give the original image. ``0.0`` makes the image to be blurred.
            ``2.0`` will enhance the sharpness by a factor of 2.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `PIL/ImageEnhance <https://pillow.readthedocs.io/en/stable/reference/ImageEnhance.html>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
        - `Tensorflow Model augment <https://github.com/tensorflow/models/blob/v2.12.0/official/vision/ops/augment.py>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
        factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        self.factor = augmentation_utils.parse_factor(
            factor, max_value=None, center_value=1, seed=seed
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        return self.factor(
            shape=(batch_size, 1, 1, 1), dtype=self.compute_dtype
        )

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        new_transformation = tf.expand_dims(transformation, axis=0)
        output = self.augment_images(images, new_transformation)
        return tf.squeeze(output, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = preprocessing_utils.transform_value_range(
            images,
            original_range=self.value_range,
            target_range=(0, 255),
            dtype=self.compute_dtype,
        )
        original_images = images
        # [1 1 1]
        # [1 5 1]
        # [1 1 1]
        # all divided by 13 is the default 3x3 gaussian smoothing kernel.
        # Correlating or Convolving with this filter is equivalent to performing
        # a gaussian blur.
        kernel = (
            tf.constant(
                [[1, 1, 1], [1, 5, 1], [1, 1, 1]],
                dtype=self.compute_dtype,
                shape=[3, 3, 1, 1],
            )
            / 13.0
        )
        # Tile across channel dimension.
        channels = tf.shape(images)[-1]
        kernel = tf.tile(kernel, [1, 1, channels, 1])
        strides = [1, 1, 1, 1]
        smoothed_image = tf.nn.depthwise_conv2d(
            images, kernel, strides, padding="VALID", dilations=[1, 1]
        )
        smoothed_image = tf.clip_by_value(smoothed_image, 0.0, 255.0)

        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(smoothed_image)
        mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
        smoothed_image = tf.pad(
            smoothed_image, [[0, 0], [1, 1], [1, 1], [0, 0]]
        )

        images = tf.where(tf.equal(mask, 1), smoothed_image, original_images)
        # Blend the final result.
        images = augmentation_utils.blend(
            images, original_images, transformations, (0, 255)
        )
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
                "factor": self.factor,
                "seed": self.seed,
            }
        )
        return config
