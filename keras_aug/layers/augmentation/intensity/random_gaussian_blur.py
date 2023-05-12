import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomGaussianBlur(VectorizedBaseRandomLayer):
    """Applies a Gaussian Blur with random strength to an image.

    Args:
        kernel_size (int|Sequence[int]): The x and y dimensions for the kernel
            used.
        factor (float|Sequence[float]|keras_aug.FactorSampler): The range of the
            factor that controls the extent to which the image is blurred.
            When represented as a single float, the factor will be picked
            between ``[0.0, 0.0 + upper]``. Mathematically, ``factor``
            represents the sigma value in a gaussian blur. ``0.0`` makes this
            layer perform a no-op operation. High values make the blur
            stronger.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, kernel_size, factor, seed=None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kernel_size, (tuple, list)):
            self.x = kernel_size[0]
            self.y = kernel_size[1]
        else:
            if isinstance(kernel_size, int):
                self.x = self.y = kernel_size
            else:
                raise ValueError(
                    "`kernel_size` must be list, tuple or integer "
                    ", got {} ".format(type(self.kernel_size))
                )
        self.kernel_size = kernel_size
        if isinstance(factor, (int, float)):
            factor = (0.0, factor)
        self.factor = augmentation_utils.parse_factor(
            factor, min_value=0.0, max_value=None, seed=seed
        )

    def get_random_transformation_batch(self, batch_size, **kwargs):
        factors = self.factor(shape=(batch_size, 1))
        blur_vs = self.get_kernel(factors, self.y)
        blur_hs = self.get_kernel(factors, self.x)
        blur_vs = tf.reshape(blur_vs, [batch_size, self.y, 1, 1, 1])
        blur_hs = tf.reshape(blur_hs, [batch_size, 1, self.x, 1, 1])
        return {"blur_vs": blur_vs, "blur_hs": blur_hs}

    def augment_ragged_image(self, image, transformation, **kwargs):
        images = tf.expand_dims(image, axis=0)
        transformations = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        images = self.augment_images(
            images=images, transformations=transformations, **kwargs
        )
        return tf.squeeze(images, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = tf.cast(images, dtype=self.compute_dtype)
        num_channels = tf.shape(images)[-1]
        blur_vs = transformations["blur_vs"]
        blur_hs = transformations["blur_hs"]
        blur_vs = tf.tile(blur_vs, [1, 1, 1, num_channels, 1])
        blur_vs = tf.cast(blur_vs, dtype=self.compute_dtype)
        blur_hs = tf.tile(blur_hs, [1, 1, 1, num_channels, 1])
        blur_hs = tf.cast(blur_hs, dtype=self.compute_dtype)
        inputs_for_gaussian_blur_single_image = {
            "images": images,
            "blur_vs": blur_vs,
            "blur_hs": blur_hs,
        }
        images = tf.vectorized_map(
            self.gaussian_blur_single_image,
            inputs_for_gaussian_blur_single_image,
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

    def get_kernel(self, factor, filter_size):
        # We are running this in float32, regardless of layer's
        # self.compute_dtype. Calculating blur_filter in lower precision will
        # corrupt the final results.
        x = tf.cast(
            tf.range(-filter_size // 2 + 1, filter_size // 2 + 1),
            dtype=tf.float32,
        )
        x = tf.reshape(x, shape=(1, -1))
        blur_filter = tf.exp(
            -tf.pow(x, 2.0)
            / (2.0 * tf.pow(tf.cast(factor, dtype=tf.float32), 2.0))
        )
        blur_filter /= tf.reduce_sum(blur_filter, axis=-1, keepdims=True)
        return blur_filter

    def gaussian_blur_single_image(self, inputs):
        image = tf.expand_dims(inputs.get("images", None), axis=0)
        blur_v = inputs.get("blur_vs", None)
        blur_h = inputs.get("blur_hs", None)
        image = tf.nn.depthwise_conv2d(
            image, blur_h, strides=[1, 1, 1, 1], padding="SAME"
        )
        image = tf.nn.depthwise_conv2d(
            image, blur_v, strides=[1, 1, 1, 1], padding="SAME"
        )
        return tf.squeeze(image, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor, "kernel_size": self.kernel_size})
        return config
