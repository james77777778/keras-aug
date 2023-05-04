import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="keras_aug")
class FactorSampler:
    """FactorSampler represents a strength factor for use in an augmentation
    layer.

    FactorSampler should be subclassed and implement a `__call__()` method that
    returns a tf.float32, or a float. This method will be used by preprocessing
    layers to determine the strength of their augmentation. The specific range
    of values supported may vary by layer, but for most layers is the range
    [0, 1].

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __call__(self, shape=None, dtype=tf.float32):
        raise NotImplementedError(
            "FactorSampler subclasses must implement a `__call__()` method."
        )

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
