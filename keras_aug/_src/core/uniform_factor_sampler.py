import keras

from keras_aug._src.core.factor_sampler import FactorSampler
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.core"])
@keras.saving.register_keras_serializable(package="keras_aug")
class UniformFactorSampler(FactorSampler):
    """UniformFactorSampler samples factors uniformly from a range.

    Args:
        lower (float): the lower bound of values returned from `__call__()`.
        upper (float): the upper bound of values returned from `__call__()`.
        seed (int|float, optional): The random seed. Defaults to `None`.
    """

    def __init__(self, lower, upper, seed=None):
        super().__init__(seed=seed)
        self.lower = lower
        self.upper = upper
        self.seed = seed

    def __call__(self, shape=(), dtype="float32"):
        ops = self.backend
        return ops.random.uniform(
            shape,
            minval=self.lower,
            maxval=self.upper,
            dtype=dtype,
            seed=self.random_generator,
        )

    def get_config(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "seed": self.seed,
        }
