import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import CUSTOM_ANNOTATIONS
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import KEYPOINTS
from keras_aug.utils.augmentation import LABELS
from keras_aug.utils.augmentation import SEGMENTATION_MASKS


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomApply(VectorizedBaseRandomLayer):
    """Apply randomly an augmentation or a list of augmentations with a given
    probability.

    Notes:
        The shape and type of the outputs must be the same of the inputs.

    Args:
        layer (VectorizedBaseRandomLayer|keras.Layer|keras.Sequential): This
            layer will be applied to the batch when the sampled
            ``prob < rate``. Layer should not modify the shape of the inputs.
        rate (float, optional): The value that controls the frequency of
            applying the layer. ``1.0`` means the ``layer`` will always apply.
            ``0.0`` means no op. Defaults to ``0.5``.
        batchwise (bool, optional): Whether to pass entire batches to the
            underlying layer. When set to ``True``, each batch is passed to a
            single layer, instead of each sample to an independent layer. This
            is useful when using ``MixUp()``, ``CutMix()``, ``Mosaic()``, etc.
            Defaults to ``False``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, layer, rate=0.5, batchwise=False, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if not (0 <= rate <= 1.0):
            raise ValueError(
                f"rate must be in range [0, 1]. Received rate: {rate}"
            )
        self.layer = layer
        self.rate = rate
        self.batchwise = batchwise
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        prob = self._random_generator.random_uniform(shape=(batch_size,))
        return prob

    def compute_inputs_signature(self, inputs):
        fn_output_signature = {}
        if IMAGES in inputs:
            if isinstance(inputs[IMAGES], tf.Tensor):
                fn_output_signature[IMAGES] = tf.TensorSpec(
                    inputs[IMAGES].shape[1:], self.compute_dtype
                )
            else:
                fn_output_signature[IMAGES] = tf.RaggedTensorSpec(
                    shape=inputs[IMAGES].shape[1:],
                    ragged_rank=1,
                    dtype=self.compute_dtype,
                )
        if LABELS in inputs:
            fn_output_signature[LABELS] = tf.TensorSpec(
                inputs[LABELS].shape[1:], self.compute_dtype
            )
        if BOUNDING_BOXES in inputs:
            fn_output_signature[BOUNDING_BOXES] = {
                "boxes": tf.RaggedTensorSpec(
                    shape=[None, 4],
                    ragged_rank=1,
                    dtype=self.compute_dtype,
                ),
                "classes": tf.RaggedTensorSpec(
                    shape=[None], ragged_rank=0, dtype=self.compute_dtype
                ),
            }
        if SEGMENTATION_MASKS in inputs:
            if isinstance(inputs[SEGMENTATION_MASKS], tf.Tensor):
                fn_output_signature[SEGMENTATION_MASKS] = tf.TensorSpec(
                    inputs[SEGMENTATION_MASKS].shape[1:], self.compute_dtype
                )
            else:
                fn_output_signature[SEGMENTATION_MASKS] = tf.RaggedTensorSpec(
                    shape=inputs[SEGMENTATION_MASKS].shape[1:],
                    ragged_rank=1,
                    dtype=self.compute_dtype,
                )
        if KEYPOINTS in inputs:
            if isinstance(inputs[KEYPOINTS], tf.Tensor):
                fn_output_signature[KEYPOINTS] = tf.TensorSpec(
                    inputs[KEYPOINTS].shape[1:], self.compute_dtype
                )
            else:
                fn_output_signature[KEYPOINTS] = tf.RaggedTensorSpec(
                    shape=inputs[KEYPOINTS].shape[1:],
                    ragged_rank=1,
                    dtype=self.compute_dtype,
                )
        if CUSTOM_ANNOTATIONS in inputs:
            raise NotImplementedError()
        return fn_output_signature

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
        batch_size = tf.shape(images)[0]
        probs = self.get_random_transformation_batch(batch_size)
        if self.batchwise:
            result = self.layer(inputs) if probs[0] < self.rate else inputs
        else:
            bounding_boxes = inputs.get(BOUNDING_BOXES, None)
            # make bounding_boxes to dense first
            if bounding_boxes is not None:
                ori_bbox_info = bounding_box.validate_format(bounding_boxes)
                inputs[BOUNDING_BOXES] = bounding_box.to_dense(bounding_boxes)

            inputs_for_augment = {"inputs": inputs, "probs": probs}
            result = tf.map_fn(
                self.augment,
                inputs_for_augment,
                fn_output_signature=self.compute_inputs_signature(inputs),
            )

            bounding_boxes = result.get(BOUNDING_BOXES, None)
            if bounding_boxes is not None:
                if ori_bbox_info["ragged"]:
                    bounding_boxes = bounding_box.to_ragged(bounding_boxes)
                else:
                    bounding_boxes = bounding_box.to_dense(bounding_boxes)
                result[BOUNDING_BOXES] = bounding_boxes
        return result

    def augment(self, inputs):
        input = inputs.get("inputs", None)
        prob = inputs.get("probs", None)
        if prob < self.rate:
            result = self.layer(input)
        else:
            result = input
        if BOUNDING_BOXES in result:
            result[BOUNDING_BOXES] = bounding_box.to_ragged(
                result[BOUNDING_BOXES], dtype=self.compute_dtype
            )
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer": self.layer,
                "rate": self.rate,
                "batchwise": self.batchwise,
                "seed": self.seed,
            }
        )
        return config
