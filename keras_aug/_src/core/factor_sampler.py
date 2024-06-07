import keras
from keras import backend

from keras_aug._src.backend.dynamic_backend import DynamicBackend
from keras_aug._src.backend.dynamic_backend import DynamicRandomGenerator
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.core"])
@keras.saving.register_keras_serializable(package="keras_aug")
class FactorSampler:
    """FactorSampler represents a strength factor for use in an augmentation
    layer.

    FactorSampler should be subclassed and implement a `__call__()` method that
    returns a "float32", or a float. This method will be used by preprocessing
    layers to determine the strength of their augmentation. The specific range
    of values supported may vary by layer, but for most layers is the range
    [0, 1].
    """

    def __init__(self, has_generator=True, seed=None):
        self._backend = DynamicBackend(backend.backend())
        if has_generator:
            self._random_generator = DynamicRandomGenerator(
                backend.backend(), seed=seed
            )

    @property
    def backend(self):
        return self._backend.backend

    @property
    def random_generator(self):
        return self._random_generator.random_generator

    def __call__(self, shape=(), dtype="float32"):
        raise NotImplementedError(
            "FactorSampler subclasses must implement a `__call__()` method."
        )

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
