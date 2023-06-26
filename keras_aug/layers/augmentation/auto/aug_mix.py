import tensorflow as tf
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug import layers
from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils.augmentation import H_AXIS
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import W_AXIS
from keras_aug.utils.distribution import stateless_random_beta
from keras_aug.utils.distribution import stateless_random_dirichlet


@keras.utils.register_keras_serializable(package="keras_aug")
class AugMix(VectorizedBaseRandomLayer):
    """Performs the AugMix data augmentation technique.

    AugMix aims to produce images with variety while preserving the image
    semantics and local statistics. During the augmentation process, each image
    is augmented ``num_chains`` different ways, each way consisting of
    ``chain_depth`` augmentations. Augmentations are sampled from the list:
    [translation, shearing, rotation, posterization, histogram equalization,
    solarization and auto contrast]. The results of each chain are then mixed
    together with the original image based on random samples from a Dirichlet
    distribution.

    Args:
        value_range (Sequence[float]): The range of values the incoming
            images will have. This is typically either ``[0, 1]`` or
            ``[0, 255]`` depending on how your preprocessing pipeline is set up.
        severity (float|(float, float)|keras_aug.FactorSampler, optional): The range
            of the strength of augmentations. When represented as a single float,
            the factor will be picked between ``[0.01, upper]``. Defaults to
            ``[0.01, 0.3]``.
        num_chains (int, optional): The number of different chains to be mixed.
            Defaults to ``3``.
        chain_depth (int, Sequence[int], optional): The range of the number of
            transformations in the chains. Defaults to ``[1, 3]``.
        alpha (float, optional): The probability coefficients for the Beta and
            Dirichlet distributions. Defaults to ``1.0``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `AugMix <https://arxiv.org/abs/1912.02781>`_
        - `AugMix Official Repo <https://github.com/google-research/augmix>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(
        self,
        value_range,
        severity=[0.01, 0.3],
        num_chains=3,
        chain_depth=[1, 3],
        alpha=1.0,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        self.value_range = value_range
        if isinstance(severity, (int, float)):
            severity = (0.01, severity)
        self.severity = augmentation_utils.parse_factor(
            severity,
            min_value=0.01,
            max_value=1.0,
            seed=seed,
        )
        self.num_chains = num_chains
        if isinstance(chain_depth, int):
            chain_depth = [chain_depth, chain_depth]
        self.chain_depth = chain_depth
        self.alpha = alpha
        self.seed = seed

        # initialize layers
        self.auto_contrast = layers.AutoContrast(
            value_range=self.value_range, dtype=self.compute_dtype
        )
        self.equalize = layers.Equalize(
            value_range=self.value_range, dtype=self.compute_dtype
        )

    def get_random_transformation_batch(self, batch_size, **kwargs):
        # cast to float32 to avoid numerical issue
        # sample from dirichlet
        alpha = tf.ones([self.num_chains], dtype=tf.float32) * self.alpha
        chain_mixing_weights = stateless_random_dirichlet(
            (batch_size, self.num_chains),
            seed=self._random_generator.make_seed_for_stateless_op(),
            alpha=alpha,
            dtype=tf.float32,
        )
        # sample from beta
        weight_sample = stateless_random_beta(
            (batch_size, 1),
            seed_alpha=self._random_generator.make_seed_for_stateless_op(),
            seed_beta=self._random_generator.make_seed_for_stateless_op(),
            alpha=self.alpha,
            beta=self.alpha,
            dtype=tf.float32,
        )
        return {
            "chain_mixing_weights": chain_mixing_weights,
            "weight_sample": weight_sample,
        }

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
        original_shape = images.shape
        images = preprocessing_utils.transform_value_range(
            images, self.value_range, (0, 255), dtype=self.compute_dtype
        )
        inputs_for_aug_mix_single_image = {
            IMAGES: images,
            "transformations": transformations,
        }
        images = tf.map_fn(
            self.aug_mix_single_image,
            inputs_for_aug_mix_single_image,
            fn_output_signature=self.compute_dtype,
        )
        images = preprocessing_utils.transform_value_range(
            images, (0, 255), self.value_range, self.compute_dtype
        )
        images = tf.ensure_shape(images, shape=original_shape)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def aug_mix_single_image(self, inputs):
        image = inputs.get(IMAGES, None)
        transformation = inputs.get("transformations", None)

        chain_mixing_weights = tf.cast(
            transformation["chain_mixing_weights"], dtype=self.compute_dtype
        )
        weight_sample = tf.cast(
            transformation["weight_sample"], dtype=self.compute_dtype
        )

        result = tf.zeros_like(image, dtype=image.dtype)
        curr_chain = tf.constant([0], dtype=tf.int32)
        image, chain_mixing_weights, curr_chain, result = tf.while_loop(
            lambda image, chain_mixing_weights, curr_chain, result: tf.less(
                curr_chain, self.num_chains
            ),
            self.loop_on_width,
            [image, chain_mixing_weights, curr_chain, result],
        )
        result = weight_sample * image + (1 - weight_sample) * result
        return result

    def loop_on_width(self, image, chain_mixing_weights, curr_chain, result):
        image_aug = tf.identity(image)
        chain_depth = self._random_generator.random_uniform(
            shape=(),
            minval=self.chain_depth[0],
            maxval=self.chain_depth[1] + 1,
            dtype=tf.int32,
        )

        depth_level = tf.constant([0], dtype=tf.int32)
        depth_level, image_aug = tf.while_loop(
            lambda depth_level, image_aug: tf.less(depth_level, chain_depth),
            self.loop_on_depth,
            [depth_level, image_aug],
        )
        result += tf.gather(chain_mixing_weights, curr_chain) * image_aug
        curr_chain += 1
        return image, chain_mixing_weights, curr_chain, result

    def loop_on_depth(self, depth_level, image_aug):
        op_idx = self._random_generator.random_uniform(
            shape=(), minval=0, maxval=9, dtype=tf.int32
        )
        image_aug = self.apply_op(image_aug, op_idx)
        depth_level += 1
        return depth_level, image_aug

    def apply_op(self, image_aug, op_idx):
        augmented = image_aug
        augmented = tf.cond(
            op_idx == tf.constant([0], dtype=tf.int32),
            lambda: self.auto_contrast(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([1], dtype=tf.int32),
            lambda: self.equalize(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([2], dtype=tf.int32),
            lambda: self.posterize(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([3], dtype=tf.int32),
            lambda: self.rotate(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([4], dtype=tf.int32),
            lambda: self.solarize(augmented),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([5], dtype=tf.int32),
            lambda: self.shear(augmented, along_x=True),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([6], dtype=tf.int32),
            lambda: self.shear(augmented, along_x=False),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([7], dtype=tf.int32),
            lambda: self.translate(augmented, along_x=True),
            lambda: augmented,
        )
        augmented = tf.cond(
            op_idx == tf.constant([8], dtype=tf.int32),
            lambda: self.translate(augmented, along_x=False),
            lambda: augmented,
        )
        return augmented

    def posterize(self, image):
        ori_dtype = image.dtype
        bits = tf.cast(self.severity() * 3, tf.int32)
        shift = tf.cast(4 - bits + 1, tf.uint8)
        image = tf.cast(image, tf.uint8)
        image = tf.bitwise.left_shift(
            tf.bitwise.right_shift(image, shift), shift
        )
        image = tf.cast(image, dtype=ori_dtype)
        return image

    def rotate(self, image):
        angle = tf.expand_dims(
            self.severity(shape=(1,), dtype=tf.float32) * 30, axis=0
        )
        height = tf.expand_dims(tf.shape(image)[H_AXIS : H_AXIS + 1], axis=0)
        width = tf.expand_dims(tf.shape(image)[W_AXIS : W_AXIS + 1], axis=0)
        height = tf.cast(height, dtype=tf.float32)
        width = tf.cast(width, dtype=tf.float32)
        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if image.dtype == tf.bfloat16:
            image = tf.cast(image, dtype=tf.float32)
        image = preprocessing_utils.transform(
            tf.expand_dims(image, axis=0),
            augmentation_utils.get_rotation_matrix(angle, height, width),
        )
        image = tf.squeeze(image, axis=0)
        return tf.cast(image, dtype=self.compute_dtype)

    def solarize(self, image):
        threshold = tf.cast(
            tf.cast(self.severity() * 255, tf.int32), image.dtype
        )
        image = tf.where(image < threshold, image, 255 - image)
        return image

    def shear(self, image, along_x=True):
        factor = tf.cast(self.severity() * 0.3, tf.float32)
        factor *= preprocessing_utils.random_inversion(self._random_generator)
        if along_x:
            transform = tf.convert_to_tensor(
                [1.0, factor, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            )
        else:
            transform = tf.convert_to_tensor(
                [1.0, 0.0, 0.0, factor, 1.0, 0.0, 0.0, 0.0]
            )
        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if image.dtype == tf.bfloat16:
            image = tf.cast(image, dtype=tf.float32)
        image = preprocessing_utils.transform(
            tf.expand_dims(image, axis=0),
            tf.expand_dims(transform, axis=0),
        )
        image = tf.squeeze(image, axis=0)
        return tf.cast(image, dtype=self.compute_dtype)

    def translate(self, image, along_x=True):
        shape = tf.cast(tf.shape(image), tf.float32)
        if along_x:
            size = shape[1]
        else:
            size = shape[0]
        factor = tf.cast(self.severity() * size / 3, tf.float32)
        factor *= preprocessing_utils.random_inversion(self._random_generator)
        if along_x:
            transform = tf.convert_to_tensor(
                [1.0, 0.0, factor, 0.0, 1.0, 0.0, 0.0, 0.0]
            )
        else:
            transform = tf.convert_to_tensor(
                [1.0, 0.0, 0.0, 0.0, 1.0, factor, 0.0, 0.0]
            )
        # tf.raw_ops.ImageProjectiveTransformV3 not support bfloat16
        if image.dtype == tf.bfloat16:
            image = tf.cast(image, dtype=tf.float32)
        image = preprocessing_utils.transform(
            tf.expand_dims(image, axis=0),
            tf.expand_dims(transform, axis=0),
        )
        image = tf.squeeze(image, axis=0)
        return tf.cast(image, dtype=self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value_range": self.value_range,
                "severity": self.severity,
                "num_chains": self.num_chains,
                "chain_depth": self.chain_depth,
                "alpha": self.alpha,
                "seed": self.seed,
            }
        )
        return config
