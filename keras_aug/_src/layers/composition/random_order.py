import typing

import keras
from keras import backend
from keras import saving
from keras.src.utils.backend_utils import in_tf_graph

from keras_aug._src.backend.dynamic_backend import DynamicBackend
from keras_aug._src.backend.dynamic_backend import DynamicRandomGenerator
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(
    parent_path=["keras_aug.layers.composition", "keras_aug.layers"]
)
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomOrder(keras.Layer):
    """Apply a list of transformations in a random order.

    Note that due to implementation limitations, a single sampled ordering
    will be applied to the entire batch of inputs.

    Args:
        transforms: A list of transformations or a `keras.Layer`.
    """

    def __init__(self, transforms, seed=None, **kwargs):
        super().__init__(**kwargs)
        self._backend = DynamicBackend(backend.backend())
        self._random_generator = DynamicRandomGenerator(
            backend.backend(), seed=seed
        )
        self.seed = seed

        # Check
        if not isinstance(transforms, (typing.Sequence, keras.Layer)):
            raise ValueError(
                "`transforms` must be a sequence (e.g. tuple and list) or a "
                "`keras.Layer`. "
                f"Received: transforms={transforms} of type {type(transforms)}"
            )
        if isinstance(transforms, keras.Layer):
            transforms = [transforms]
        self.transforms = list(transforms)
        self.total = len(self.transforms)

        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True
        self.autocast = False

    @property
    def backend(self):
        return self._backend.backend

    @property
    def random_generator(self):
        return self._random_generator.random_generator

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        for transfrom in self.transforms:
            output_shape = transfrom.compute_output_shape(output_shape)
        return output_shape

    def get_params(self):
        ops = self.backend
        random_generator = self.random_generator

        fn_idx = ops.random.shuffle(
            ops.numpy.arange(self.total, dtype="int32"), seed=random_generator
        )
        return fn_idx

    def __call__(self, inputs, **kwargs):
        if in_tf_graph():
            self._set_backend("tensorflow")
            try:
                outputs = super().__call__(inputs, **kwargs)
            finally:
                self._reset_backend()
            return outputs
        else:
            return super().__call__(inputs, **kwargs)

    def call(self, inputs):
        ops = self.backend
        fn_idx = self.get_params()

        outputs = inputs
        for i in range(self.total):
            idx = fn_idx[i]
            outputs = ops.core.switch(idx, self.transforms, outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "transforms": saving.serialize_keras_object(self.transforms),
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = config.copy()
        config["transforms"] = saving.deserialize_keras_object(
            config["transforms"], custom_objects=custom_objects
        )
        return cls(**config)

    def _set_backend(self, name):
        self._backend.set_backend(name)
        self._random_generator.set_generator(name)

    def _reset_backend(self):
        self._backend.reset()
        self._random_generator.reset()
