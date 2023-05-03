import tensorflow as tf
from keras_cv.utils import fill_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import BATCHED
from keras_aug.utils.augmentation import H_AXIS
from keras_aug.utils.augmentation import LABELS
from keras_aug.utils.augmentation import W_AXIS


@keras.utils.register_keras_serializable(package="keras_aug")
class CutMix(VectorizedBaseRandomLayer):
    """CutMix implements the CutMix data augmentation technique.

    CutMix only supports dense images as inputs.

    Args:
        alpha (float, optional): The inverse scale parameter between 0 to +inf
            for the gamma distribution. This controls the shape of the
            distribution from which the smoothing values are sampled.
            Defaults to ``1.0``, which is a recommended value when training an
            ImageNet classification model.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `CutMix <https://arxiv.org/abs/1905.04899>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(
        self,
        alpha=1.0,
        seed=None,
        **kwargs,
    ):
        # CutMix layer uses stateless rng generator for following random
        # operations
        super().__init__(seed=seed, force_generator=True, **kwargs)
        self.alpha = alpha
        self.seed = seed

        # set force_no_unwrap_ragged_image_call=True because MosaicYOLOV8 needs
        # to process images in batch.
        self.force_no_unwrap_ragged_image_call = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        height = tf.cast(tf.shape(images)[H_AXIS], dtype=self.compute_dtype)
        width = tf.cast(tf.shape(images)[W_AXIS], dtype=self.compute_dtype)
        permutation_order = self._random_generator.random_uniform(
            (batch_size,),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32,
        )
        lambda_samples = self.sample_from_beta(
            self.alpha, self.alpha, (batch_size,)
        )
        lambda_samples = tf.cast(lambda_samples, dtype=self.compute_dtype)
        ratios = tf.math.sqrt(1.0 - lambda_samples)
        height = tf.cast(tf.shape(images)[H_AXIS], dtype=self.compute_dtype)
        width = tf.cast(tf.shape(images)[W_AXIS], dtype=self.compute_dtype)
        cut_heights = tf.cast(ratios * height, dtype=tf.int32)
        cut_widths = tf.cast(ratios * width, dtype=tf.int32)
        center_xs = self._random_generator.random_uniform(
            shape=(batch_size,), dtype=self.compute_dtype
        )
        center_xs = tf.cast(center_xs * width, dtype=tf.int32)
        center_ys = self._random_generator.random_uniform(
            shape=(batch_size,), dtype=self.compute_dtype
        )
        center_ys = tf.cast(center_ys * height, dtype=tf.int32)
        return {
            "permutation_order": permutation_order,
            "center_xs": center_xs,
            "center_ys": center_ys,
            "cut_heights": cut_heights,
            "cut_widths": cut_widths,
        }

    def augment_images(self, images, transformations, **kwargs):
        if isinstance(images, tf.RaggedTensor):
            raise ValueError(
                "CutMix expects dense images. Received: images type: "
                f"{type(images)}"
            )
        permutation_order = transformations["permutation_order"]
        center_xs = transformations["center_xs"]
        center_ys = transformations["center_ys"]
        cut_heights = transformations["cut_heights"]
        cut_widths = transformations["cut_widths"]
        images = fill_utils.fill_rectangle(
            images,
            center_xs,
            center_ys,
            cut_widths,
            cut_heights,
            tf.gather(images, permutation_order),
        )
        return images

    def augment_labels(self, labels, transformations, images=None, **kwargs):
        labels = tf.cast(labels, dtype=self.compute_dtype)
        height = tf.cast(tf.shape(images)[H_AXIS], dtype=self.compute_dtype)
        width = tf.cast(tf.shape(images)[W_AXIS], dtype=self.compute_dtype)
        permutation_order = transformations["permutation_order"]
        cutmix_labels = tf.gather(labels, permutation_order)

        cut_heights = tf.cast(
            transformations["cut_heights"], dtype=self.compute_dtype
        )
        cut_widths = tf.cast(
            transformations["cut_widths"], dtype=self.compute_dtype
        )
        bounding_box_area = cut_heights * cut_widths
        lambda_sample = 1.0 - bounding_box_area / (height * width)
        lambda_sample = tf.reshape(lambda_sample, [-1, 1])

        labels = lambda_sample * labels + (1.0 - lambda_sample) * cutmix_labels
        return labels

    def _batch_augment(self, inputs):
        self._validate_inputs(inputs)
        return super()._batch_augment(inputs)

    def call(self, inputs):
        _, metadata = self._format_inputs(inputs)
        if metadata[BATCHED] is not True:
            raise ValueError(
                "CutMix received a single image to `call`. The "
                "layer relies on combining multiple examples, and as such "
                "will not behave as expected. Please call the layer with 2 "
                "or more samples."
            )
        return super().call(inputs=inputs)

    def _validate_inputs(self, inputs):
        labels = inputs.get(LABELS, None)
        if labels is None:
            raise ValueError(
                "CutMix expects 'labels' to be present in its inputs. "
                "CutMix relies on both images an labels. "
                "Please pass a dictionary with keys 'images' "
                "containing the image Tensor, and 'labels' containing "
                "the classification labels. "
                "For example, `cut_mix({'images': images, 'labels': labels})`."
            )

    def sample_from_beta(self, alpha, beta, shape):
        sample_alpha = tf.random.stateless_gamma(
            shape,
            alpha=alpha,
            seed=self._random_generator.make_seed_for_stateless_op(),
        )
        sample_beta = tf.random.stateless_gamma(
            shape,
            alpha=beta,
            seed=self._random_generator.make_seed_for_stateless_op(),
        )
        return sample_alpha / (sample_alpha + sample_beta)

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "seed": self.seed})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
