import tensorflow as tf
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from tensorflow import keras

from keras_aug.utils import augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomBlur(VectorizedBaseImageAugmentationLayer):
    """Blur the images using random-sized kernels.

    This layer applies a mean filter with varying kernel sizes to blur the
    images.

    Args:
        blur_limit: A tuple of int or an int represents kernel size range for
            blurring the input image. Should be in range [3, inf).
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(
        self,
        blur_limit,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.blur_limit = augmentation_utils.to_tuple(blur_limit, low=3)

    def get_random_transformation_batch(self, batch_size, **kwargs):
        bias = self.blur_limit[0]
        num_blur_limit = (self.blur_limit[1] - self.blur_limit[0]) // 2
        blur_kernel_sizes = self._random_generator.random_uniform(
            shape=(batch_size, 1),
            minval=0,
            maxval=num_blur_limit + 1,
            dtype=tf.int32,
        )
        # [0, k] => [0, ..., 2k+1] ensures only odd numbers
        blur_kernel_sizes = blur_kernel_sizes * 2 + bias
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
        blurred_images = tf.map_fn(
            self.blur_single_image,
            inputs_for_blur_single_image,
            fn_output_signature=tf.float32,
        )
        return tf.cast(blurred_images, self.compute_dtype)

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
        config.update({"blur_limit": self.blur_limit, "seed": self.seed})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
