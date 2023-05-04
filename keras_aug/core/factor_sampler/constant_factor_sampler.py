import tensorflow as tf
from tensorflow import keras

from keras_aug.core.factor_sampler.factor_sampler import FactorSampler


@keras.utils.register_keras_serializable(package="keras_aug")
class ConstantFactorSampler(FactorSampler):
    """ConstantFactorSampler samples the same factor for every call to
    ``__call__()``.

    Args:
        value (int|float): the value to return from ``__call__()``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, value):
        self.value = value

    def __call__(self, shape=(), dtype="float32"):
        return tf.ones(shape=shape, dtype=dtype) * self.value

    def get_config(self):
        return {"value": self.value}
