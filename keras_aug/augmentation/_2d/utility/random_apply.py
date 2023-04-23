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

    RandomApply applies augmentation(s) defined by ``layer`` in batch. For
    example, if the sampled probability=0.6 and ``rate=0.5`` then no op for
    this entire batch. The inputs must be dense tensor and the ``layer`` should
    not modify the shape of the inputs.

    Args:
        layer (VectorizedBaseRandomLayer|keras.Layer|keras.Sequential): This
            layer will be applied to the batch when the sampled
            ``prob < rate``. Layer should not modify the shape of the inputs.
        rate (float, optional): The value that controls the frequency of
            applying the layer. ``1.0`` means the ``layer`` will always apply.
            ``0.0`` means no op. Defaults to ``0.5``.
        seed (int|float, optional): The random seed. Defaults to
            ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
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
        prob = self._random_generator.random_uniform(shape=(1,))
        return prob

    def _batch_augment(self, inputs):
        images = inputs.get(IMAGES, None)
        batch_size = tf.shape(images)[0]
        prob = self.get_random_transformation_batch(batch_size)
        if prob < self.rate:
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
