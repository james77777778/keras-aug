import typing

import keras

from keras_aug._src.core.factor_sampler import FactorSampler
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.core"])
@keras.saving.register_keras_serializable(package="keras_aug")
class ConstantFactorSampler(FactorSampler):
    """ConstantFactorSampler samples the same factor for every call to
    `__call__()`.

    Args:
        value: The value to return from `__call__()`.
    """

    def __init__(self, value: typing.Union[int, float]):
        super().__init__(has_generator=False)
        self.value = value

    def __call__(self, shape=(), dtype="float32"):
        ops = self.backend
        return ops.numpy.ones(shape=shape, dtype=dtype) * self.value

    def get_config(self):
        return {"value": self.value}
