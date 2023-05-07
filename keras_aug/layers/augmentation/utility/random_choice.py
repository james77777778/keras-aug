from functools import partial

import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import IMAGES


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomChoice(VectorizedBaseRandomLayer):
    """RandomChoice constructs a pipeline based on provided arguments.

    The implemented policy does the following: for each input provided in
    `call`(), the policy selects a random layer from the provided list of
    `layers`. It then calls the `layer()` on the inputs.

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

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
        batch_size = tf.shape(images)[0]
        transformations = self.get_random_transformation_batch(batch_size)
        if self.batchwise:
            selected_op_idx = transformations[0]
            result = self.random_choice_single_input(
                {"inputs": inputs, "transformations": selected_op_idx}
            )
        else:
            inputs_for_random_choice_single_input = {
                "inputs": inputs,
                "transformations": transformations,
            }
            result = tf.map_fn(
                self.random_choice_single_input,
                inputs_for_random_choice_single_input,
            )
        # unpack result to normal inputs
        result = result["inputs"]
        return result

    def random_choice_single_input(self, inputs):
        input = inputs.get("inputs")
        selected_op_idx = inputs.get("transformations")
        # construct branch_fns
        branch_fns = {}
        for idx, layer in enumerate(self.layers):
            branch_fns[idx] = partial(layer, input)
        # augment
        input = tf.switch_case(selected_op_idx, branch_fns=branch_fns)
        return {"inputs": input, "transformations": selected_op_idx}

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
