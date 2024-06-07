import keras

from keras_aug._src.core.factor_sampler import FactorSampler
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.core"])
@keras.saving.register_keras_serializable(package="keras_aug")
class NormalFactorSampler(FactorSampler):
    """NormalFactorSampler samples factors from a normal distribution.

    Args:
        mean (float): The mean value for the distribution.
        stddev (float): The standard deviation of the distribution.
        min_value (float): The minimum value.
        max_value (float): The maximum value.
        seed (int|float, optional): The random seed. Defaults to ``None``.
    """

    def __init__(self, mean, stddev, min_value, max_value, seed=None):
        super().__init__(seed=seed)
        self.mean = mean
        self.stddev = stddev
        self.min_value = min_value
        self.max_value = max_value
        self.seed = seed

    def __call__(self, shape=(), dtype="float32"):
        ops = self.backend
        return ops.numpy.clip(
            ops.random.normal(
                shape=shape,
                mean=self.mean,
                stddev=self.stddev,
                dtype=dtype,
                seed=self.random_generator,
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
