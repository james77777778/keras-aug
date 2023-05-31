from functools import partial

import tensorflow as tf
from keras_cv import bounding_box
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import CUSTOM_ANNOTATIONS
from keras_aug.utils.augmentation import IMAGES
from keras_aug.utils.augmentation import KEYPOINTS
from keras_aug.utils.augmentation import LABELS
from keras_aug.utils.augmentation import SEGMENTATION_MASKS


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomChoice(VectorizedBaseRandomLayer):
    """RandomChoice constructs a pipeline based on provided arguments.

    The implemented policy does the following: for each input provided in
    `call`(), the policy selects a random layer from the provided list of
    `layers`. It then calls the `layer()` on the inputs.

    Notes:
        The shape and type of the outputs must be the same of the inputs.

    Args:
        layers (list(VectorizedBaseRandomLayer|keras.Layer|keras.Sequential)): The list
            of the layers that will be picked randomly for the pipeline.
        batchwise (bool, optional): Whether to pass entire batches to the
            underlying layer. When set to ``True``, each batch is passed to a
            single layer, instead of each sample to an independent layer. This
            is useful when using ``MixUp()``, ``CutMix()``, ``Mosaic()``, etc.
            Defaults to ``False``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(self, layers, batchwise=False, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.layers = layers
        self.batchwise = batchwise
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        selected_op_idx = self._random_generator.random_uniform(
            shape=(batch_size,),
            minval=0,
            maxval=len(self.layers),
            dtype=tf.int32,
        )
        return selected_op_idx

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
        transformations = self.get_random_transformation_batch(batch_size)
        if self.batchwise:
            selected_op_idx = transformations[0]
            result = self.augment(
                {"inputs": inputs, "transformations": selected_op_idx}
            )
        else:
            bounding_boxes = inputs.get(BOUNDING_BOXES, None)
            # make bounding_boxes to dense first
            if bounding_boxes is not None:
                ori_bbox_info = bounding_box.validate_format(bounding_boxes)
                inputs[BOUNDING_BOXES] = bounding_box.to_dense(bounding_boxes)

            inputs_for_random_choice_single_input = {
                "inputs": inputs,
                "transformations": transformations,
            }
            result = tf.map_fn(
                self.augment,
                inputs_for_random_choice_single_input,
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
        input = inputs.get("inputs")
        selected_op_idx = inputs.get("transformations")
        # construct branch_fns
        branch_fns = {}
        for idx, layer in enumerate(self.layers):
            branch_fns[idx] = partial(layer, input)
        # augment
        result = tf.switch_case(selected_op_idx, branch_fns=branch_fns)
        if BOUNDING_BOXES in result:
            result[BOUNDING_BOXES] = bounding_box.to_ragged(
                result[BOUNDING_BOXES], dtype=self.compute_dtype
            )
        result = augmentation_utils.cast_to(result, dtype=self.compute_dtype)
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers": self.layers,
                "batchwise": self.batchwise,
                "seed": self.seed,
            }
        )
        return config
