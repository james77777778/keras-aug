import tensorflow as tf
from keras_cv.core.factor_sampler.factor_sampler import FactorSampler
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_cv")
class SignedConstantFactorSampler(FactorSampler):
    """SignedConstantFactorSampler samples the same factor for every call to
    ``__call__()`` and randomly inverts the sampled factors.

    This is useful in cases where a user wants to always ensure that an
    augmentation layer performs augmentations of the same strength.

    Args:
        value: the value to return from `__call__()`.
        rate (float, optional): The rate to invert factors. Defaults to ``0.5``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    Usage:
    ```python
    constant_factor = keras_aug.core.SignedConstantFactorSampler(0.5)
    random_sharpness = keras_aug.augmentation.RandomSharpness(
        factor=constant_factor
    )
    # random_sharpness will now always use a factor of 0.5.
    # Then, the value is randomly inverted by rate.
    ```
    """

    def __init__(self, value, rate=0.5, seed=None):
        self.value = value
        self.rate = rate
        self.seed = seed

    def __call__(self, shape=(), dtype="float32"):
        factors = tf.ones(shape=shape, dtype=dtype) * self.value
        negates = tf.random.uniform(
            shape=shape, minval=0, maxval=1, dtype=tf.float32
        )
        negates = tf.cast(tf.where(negates > self.rate, -1.0, 1.0), dtype=dtype)
        return factors * negates

    def get_config(self):
        return {
            "value": self.value,
            "rate": self.rate,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
