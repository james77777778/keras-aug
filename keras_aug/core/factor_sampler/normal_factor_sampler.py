import keras
import keras.backend
import tensorflow as tf

from keras_aug.core.factor_sampler.factor_sampler import FactorSampler


@keras.utils.register_keras_serializable(package="keras_aug")
class NormalFactorSampler(FactorSampler):
    """NormalFactorSampler samples factors from a normal distribution.

    Args:
        mean (float): The mean value for the distribution.
        stddev (float): The standard deviation of the distribution.
        min_value (float): The minimum value.
        max_value (float): The maximum value.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, mean, stddev, min_value, max_value, seed=None):
        self.mean = mean
        self.stddev = stddev
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed
        self.rng = keras.backend.RandomGenerator(
            seed=seed, rng_type="stateless"
        )

    def __call__(self, shape=(), dtype="float32"):
        return tf.clip_by_value(
            self.rng.random_normal(
                shape=shape,
                mean=self.mean,
                stddev=self.stddev,
                dtype=dtype,
            ),
            self.min_value,
            self.max_value,
        )

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "seed": self.seed,
        }
