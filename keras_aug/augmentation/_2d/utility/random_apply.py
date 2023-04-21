import tensorflow as tf
from tensorflow import keras

from keras_aug.augmentation._2d.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils.augmentation import IMAGES


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomApply(VectorizedBaseRandomLayer):
    """Apply randomly an augmentation or a list of augmentations with a given
    probability.

    Currently, RandomApply applies augmentation(s) defined by `layer` in batch.
    For example, if sampled probability=0.6 and `rate=0.5` then no operation for
    this batch.

    Args:
        layer: a keras `Layer` or `VectorizedBaseRandomLayer`. This layer will
            be applied to randomly chosen samples in a batch. Layer should not
            modify the size of provided inputs.
        rate: controls the frequency of applying the layer. 1.0 means all
            elements in a batch will be modified. 0.0 means no elements will be
            modified. Defaults to 0.5.
        seed: Used to create a random seed, defaults to None.
    """

    def __init__(self, layer, rate=0.5, seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if not (0 <= rate <= 1.0):
            raise ValueError(
                f"rate must be in range [0, 1]. Received rate: {rate}"
            )
        self.layer = layer
        self.rate = rate
        self.seed = seed

    def get_random_transformation_batch(self, batch_size, **kwargs):
        probs = self._random_generator.random_uniform(shape=(1,))
        return probs

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
        batch_size = tf.shape(images)[0]
        probs = self.get_random_transformation_batch(batch_size)
        if probs < self.rate:
            result = self.layer(inputs)
        else:
            result = inputs
        return result

    def get_config(self):
        config = super().get_config()
        config.update(
            {"layer": self.layer, "rate": self.rate, "seed": self.seed}
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
