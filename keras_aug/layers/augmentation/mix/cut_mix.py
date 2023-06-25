import tensorflow as tf
from keras_cv.utils import fill_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import BATCHED
from keras_aug.utils.augmentation import H_AXIS
from keras_aug.utils.augmentation import LABELS
from keras_aug.utils.augmentation import SEGMENTATION_MASKS
from keras_aug.utils.augmentation import W_AXIS
from keras_aug.utils.distribution import stateless_random_beta


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
        super().__init__(seed=seed, **kwargs)
        self.alpha = alpha
        self.seed = seed

        # set force_no_unwrap_ragged_image_call=True because CutMix needs
        # to process images in batch.
        self.force_no_unwrap_ragged_image_call = True

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        height = tf.cast(tf.shape(images)[H_AXIS], dtype=tf.float32)
        width = tf.cast(tf.shape(images)[W_AXIS], dtype=tf.float32)
        permutation_order = self._random_generator.random_uniform(
            (batch_size,),
            minval=0,
            maxval=batch_size,
            dtype=tf.int32,
        )
        lambda_samples = stateless_random_beta(
            (batch_size,),
            seed_alpha=self._random_generator.make_seed_for_stateless_op(),
            seed_beta=self._random_generator.make_seed_for_stateless_op(),
            alpha=self.alpha,
            beta=self.alpha,
            dtype=tf.float32,
        )
        ratios = tf.math.sqrt(1.0 - lambda_samples)
        cut_heights = tf.cast(ratios * height, dtype=tf.int32)
        cut_widths = tf.cast(ratios * width, dtype=tf.int32)
        center_xs = self._random_generator.random_uniform(shape=(batch_size,))
        center_xs = tf.cast(center_xs * width, dtype=tf.int32)
        center_ys = self._random_generator.random_uniform(shape=(batch_size,))
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

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        if isinstance(segmentation_masks, tf.RaggedTensor):
            raise ValueError(
                "CutMix expects dense segmentation_masks. Received: "
                f"segmentation_masks type: {type(segmentation_masks)}"
            )
        permutation_order = transformations["permutation_order"]
        center_xs = transformations["center_xs"]
        center_ys = transformations["center_ys"]
        cut_heights = transformations["cut_heights"]
        cut_widths = transformations["cut_widths"]
        segmentation_masks = fill_utils.fill_rectangle(
            segmentation_masks,
            center_xs,
            center_ys,
            cut_widths,
            cut_heights,
            tf.gather(segmentation_masks, permutation_order),
        )
        return segmentation_masks

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
        segmentation_masks = inputs.get(SEGMENTATION_MASKS, None)
        if labels is None and segmentation_masks is None:
            raise ValueError(
                "CutMix expects `labels` or `segmentation_masks` to be present "
                "in its inputs. "
                "For example, `cut_mix({'images': images, 'labels': labels})`."
            )

    def get_config(self):
        config = super().get_config()
        config.update({"alpha": self.alpha, "seed": self.seed})
        return config
