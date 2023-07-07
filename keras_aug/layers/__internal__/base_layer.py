"""
Copied from:
https://github.com/keras-team/keras/blob/4829ddfc5fdcb84efe19b5e571f4be8dda89a3e1/keras/engine/base_layer.py

Remove __init__ docstring to prevent docstring population
"""  # noqa: E501

import tensorflow as tf
from tensorflow import keras

from keras_aug.core.keras_random_generator import KerasRandomGenerator


class BaseRandomLayer(keras.layers.Layer):
    """A layer handle the random number creation and savemodel behavior.

    Args:
          seed: optional integer, used to create RandomGenerator.
          force_generator: boolean, default to False, whether to force the
            RandomGenerator to use the code branch of tf.random.Generator.
          rng_type: string, the rng type that will be passed to backend
            RandomGenerator. Default to `None`, which will allow RandomGenerator
            to choose types by itself. Valid values are "stateful", "stateless",
            "legacy_stateful".
          **kwargs: other keyword arguments that will be passed to the parent
            *class
    """

    @tf.__internal__.tracking.no_automatic_dependency_tracking
    def __init__(
        self, seed=None, force_generator=False, rng_type=None, **kwargs
    ):
        super().__init__(**kwargs)
        self._random_generator = KerasRandomGenerator(
            seed, force_generator=force_generator, rng_type=rng_type
        )

    def build(self, input_shape):
        super().build(input_shape)
        self._random_generator._maybe_init()

    def _trackable_children(self, save_type="checkpoint", **kwargs):
        if save_type == "savedmodel":
            cache = kwargs["cache"]
            # TODO(b/213628533): This must be called before super() to ensure
            # that any input shape changes are applied before getting the config
            # of the model.
            children = self._trackable_saved_model_saver.trackable_children(
                cache
            )
            # This method exposes the self._random_generator to SavedModel only
            # (not layer.weights and checkpoint).
            children["_random_generator"] = self._random_generator
        else:
            children = {}
        children.update(super()._trackable_children(save_type, **kwargs))
        return children

    def _lookup_dependency(self, name):
        # When loading from a Keras SavedModel load, make sure that the loader
        # can find the random generator, otherwise the loader will assume that
        # it does not exist, and will try to create a new generator.
        if name == "_random_generator":
            return self._random_generator
        else:
            return super()._lookup_dependency(name)
