import tensorflow as tf
from keras_cv.core.factor_sampler.factor_sampler import FactorSampler
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_cv")
class SignedNormalFactorSampler(FactorSampler):
    """SignedNormalFactorSampler samples factors from a normal distribution and
    randomly inverts the sampled factors.

    This is useful in cases where a user wants to always ensure that an
    augmentation layer performs augmentations of the same strength.

    Args:
        mean (float): The mean value for the distribution.
        stddev (float): The standard deviation of the distribution.
        min_value (float): The values below ``min_value`` are clipped to
            ``min_value``.
        max_value (float): The values above ``max_value`` are clipped to
            ``max_value``.
        rate (float, optional): The rate to invert factors. Defaults to ``0.5``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    Usage:
    ```python
    factor = keras_aug.core.SignedNormalFactor(
        mean=0.5,
        stddev=0.1,
        lower=0,
        upper=1,
        rate=0.5,
    )
    random_sharpness = keras_aug.augmentation.RandomSharpness(factor=factor)
    # random_sharpness will now firstly sample normally around 0.5, with a
    # lower of 0 and upper bound of 1.
    # Then, the value is randomly inverted by rate.
    ```
    """

    def __init__(self, mean, stddev, min_value, max_value, rate=0.5, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.min_value = min_value
        self.max_value = max_value
        self.rate = rate
        self.seed = seed

    def __call__(self, shape=(), dtype="float32"):
        if self.stddev != 0:
            factors = tf.random.normal(
                shape=shape,
                mean=self.mean,
                stddev=self.stddev,
                seed=self.seed,
                dtype=dtype,
            )
        else:
            # if self.stddev == 0, degrade to SignedConstantFactorSampler
            factors = tf.ones(shape=shape, dtype=dtype) * self.mean
        factors = tf.clip_by_value(factors, self.min_value, self.max_value)
        negates = tf.random.uniform(
            shape=shape, minval=0, maxval=1, dtype=tf.float32
        )
        negates = tf.cast(tf.where(negates > self.rate, -1.0, 1.0), dtype=dtype)
        return factors * negates

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "rate": self.rate,
            "seed": self.seed,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)
