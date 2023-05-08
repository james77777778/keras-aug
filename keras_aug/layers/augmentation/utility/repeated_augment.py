import tensorflow as tf
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES


@keras.utils.register_keras_serializable(package="keras_aug")
class RepeatedAugment(VectorizedBaseRandomLayer):
    """RepeatedAugment augments each image in a batch multiple times.

    This technique exists to emulate the behavior of stochastic gradient descent
    within the context of mini-batch gradient descent. When training large
    vision models, choosing a large batch size can introduce too much noise into
    aggregated gradients causing the overall batch's gradients to be less
    effective than gradients produced using smaller gradients.
    RepeatedAugment handles this by re-using the same image multiple times
    within a batch creating correlated samples.

    Notes:
        This layer increases your batch size by a factor of ``len(layers)``.

    Args:
        layers (list(keras_aug.layers.*)): The list of the layers to use to
            augment the inputs.
        shuffle (bool, optional): Whether to shuffle the results. Essential when
            using an asynchronous distribution strategy such as
            ParameterServerStrategy. Defaults to ``True``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `RepeatedAugment <https://arxiv.org/abs/1901.09335>`_
        - `DEIT <https://github.com/facebookresearch/deit>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """  # noqa: E501

    def __init__(self, layers, shuffle=True, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        self.layers = layers
        self.shuffle = shuffle
        self.seed = seed

    def _batch_augment(self, inputs):
        layer_outputs = [layer(inputs) for layer in self.layers]

        results = {}
        for k in inputs.keys():
            if k == BOUNDING_BOXES:
                results[k] = {}
                results[k]["boxes"] = tf.concat(
                    [output[k]["boxes"] for output in layer_outputs], axis=0
                )
                results[k]["classes"] = tf.concat(
                    [output[k]["classes"] for output in layer_outputs], axis=0
                )
            else:
                results[k] = tf.concat(
                    [output[k] for output in layer_outputs], axis=0
                )
        if self.shuffle:
            shuffle_size = tf.shape(results[IMAGES])[0]
            results = self.shuffle_results(results, shuffle_size)
        return results

    def shuffle_results(self, results, shuffle_size):
        indices = tf.range(start=0, limit=shuffle_size, dtype=tf.int32)
        indices = tf.random.experimental.stateless_shuffle(
            indices, seed=self._random_generator.make_seed_for_stateless_op()
        )
        for k in results.keys():
            if k == BOUNDING_BOXES:
                results[k]["boxes"] = tf.gather(results[k]["boxes"], indices)
                results[k]["classes"] = tf.gather(
                    results[k]["classes"], indices
                )
            else:
                results[k] = tf.gather(results[k], indices)
        return results

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers": self.layers,
                "shuffle": self.shuffle,
                "seed": self.seed,
            }
        )
        return config
