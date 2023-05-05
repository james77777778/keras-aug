import keras
import keras.backend
import tensorflow as tf

from keras_aug.core.factor_sampler.factor_sampler import FactorSampler


@keras.utils.register_keras_serializable(package="keras_aug")
class SignedConstantFactorSampler(FactorSampler):
    """SignedConstantFactorSampler samples the same factor for every call to
    ``__call__()`` and randomly inverts the sampled factors.

    This is useful in cases where a user wants to always ensure that an
    augmentation layer performs augmentations of the same strength.

    Args:
        value: the value to return from `__call__()`.
        rate (float, optional): The rate to invert factors. Defaults to ``0.5``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, value, rate=0.5, seed=None):
        self.value = value
        self.rate = rate
        self.seed = seed
        self.rng = keras.backend.RandomGenerator(
            seed=seed, rng_type="stateless"
        )

    def __call__(self, shape=(), dtype="float32"):
        factors = tf.ones(shape=shape, dtype=dtype) * self.value
        negates = self.rng.random_uniform(
            shape=shape, minval=0, maxval=1, dtype=tf.float32
        )
        negates = tf.cast(tf.where(negates > self.rate, -1.0, 1.0), dtype=dtype)
        return factors * negates

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "value": self.value,
                "rate": self.rate,
                "seed": self.seed,
            }
        )
        return config
