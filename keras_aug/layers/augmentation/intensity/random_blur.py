import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomBlur(VectorizedBaseRandomLayer):
    """Randomly blurs the images using random-sized kernels.

    This layer applies a mean filter with varying kernel sizes to blur the
    images. The sampled kernel sizes are always odd numbers.

    Args:
        factor (int|Sequence[int]|keras_aug.FactorSampler): The kernel size range
            for blurring the input image. If the factor is a single value, the
            range will be ``(1, factor)``. The value range of the factor should
            be in ``(1, +inf)``. When sampled kernel size=``1``, there is no
            blur effect.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `Albumentations <https://github.com/albumentations-team/albumentations>`_
    """  # noqa: E501

    def __init__(
        self,
        factor,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(factor, (tuple, list)):
            factor_range = (factor[1] - factor[0]) // 2
            factor_bias = factor[0]
        else:
            factor_range = (factor[1] - 1) // 2
            factor_bias = 1
        if factor_range < 0 or factor_bias < 1:
            raise ValueError(
                "RandomBlur expects `factor` to be in range `(1, inf)`. Got: "
                f"`factor` = {factor}"
            )
        self.factor_input = factor

        self.factor_bias = factor_bias
        self.factor = preprocessing_utils.parse_factor(
            factor_range + 1, min_value=0, max_value=None, seed=seed
        )
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        blur_kernel_sizes = self.factor(shape=(batch_size, 1), dtype=tf.int32)
        # [0, k] => [0, ..., 2k+1] ensures only odd numbers
        blur_kernel_sizes = blur_kernel_sizes * 2 + self.factor_bias
        return blur_kernel_sizes

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = tf.expand_dims(transformation, axis=0)
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        inputs_for_blur_single_image = {
            "images": images,
            "blur_kernel_sizes": transformations,
        }
        images = tf.map_fn(
            self.blur_single_image,
            inputs_for_blur_single_image,
            fn_output_signature=tf.float32,
        )
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

    def blur_single_image(self, inputs):
        image = inputs.get("images", None)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32)
        blur_kernel_size = inputs.get("blur_kernel_sizes", None)
        blur_kernel_size = blur_kernel_size[0]

        mean_filter = tf.ones(
            shape=(blur_kernel_size, blur_kernel_size, 1, image.shape[-1])
        ) / tf.math.square(tf.cast(blur_kernel_size, tf.float32))
        blurred_image = tf.nn.conv2d(
            image, mean_filter, strides=1, padding="SAME"
        )
        return tf.squeeze(blurred_image, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor_input, "seed": self.seed})
        return config
