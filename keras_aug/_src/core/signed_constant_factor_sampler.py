import keras

from keras_aug._src.core.factor_sampler import FactorSampler
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.core"])
@keras.saving.register_keras_serializable(package="keras_aug")
class SignedConstantFactorSampler(FactorSampler):
    """SignedConstantFactorSampler samples the same factor for every call to
    `__call__()` and randomly inverts the sampled factors.

    This is useful in cases where a user wants to always ensure that an
    augmentation layer performs augmentations of the same strength.

    Args:
        value: the value to return from `__call__()`.
        rate (float, optional): The rate to invert factors. Defaults to `0.5`.
        seed (int|float, optional): The random seed. Defaults to `None`.
    """

    def __init__(self, value, rate=0.5, seed=None):
        super().__init__(seed=seed)
        self.value = value
        self.rate = rate
        self.seed = seed

    def __call__(self, shape=(), dtype="float32"):
        ops = self.backend
        factors = ops.numpy.ones(shape=shape, dtype=dtype) * self.value
        negates = ops.random.uniform(
            shape=shape,
            minval=0,
            maxval=1,
            dtype="float32",
            seed=self.random_generator,
        )
        negates = ops.cast(
            ops.numpy.where(negates > self.rate, -1.0, 1.0),
            dtype=dtype,
        )
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
