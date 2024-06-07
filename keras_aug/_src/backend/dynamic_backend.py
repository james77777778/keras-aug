from keras import backend
from keras import random


class DynamicBackend:
    def __init__(self, name=None):
        if name is not None and not isinstance(name, str):
            raise TypeError
        self._name = name

        # Variable
        self._backend = None

        # Init
        self.set_backend(self._name, force=True)

    @property
    def name(self):
        return self._name

    @property
    def backend(self):
        return self._backend

    def set_backend(self, name=None, force=False):
        name = name or backend.backend()
        self._backend = get_backend(name)
        self._name = name

    def reset(self):
        self.set_backend()


class DynamicRandomGenerator:
    def __init__(self, name=None, seed=None):
        if name is not None and not isinstance(name, str):
            raise TypeError
        self._name = name
        self._seed = seed

        # Variable
        self._cached_random_generator = {}

        # Init
        self.set_generator(self._name)

    @property
    def name(self):
        return self._name

    @property
    def random_generator(self):
        return self._cached_random_generator[self._name]

    def set_generator(self, name=None):
        name = name or backend.backend()
        if name in self._cached_random_generator:
            return
        self._cached_random_generator[name] = random.SeedGenerator(
            seed=self._seed, backend=get_backend(name)
        )

    def reset(self):
        self.set_generator()


def get_backend(name=None):
    name = name or backend.backend()
    if name == "tensorflow":
        import keras.src.backend.tensorflow as module
    elif name == "jax":
        import keras.src.backend.jax as module
    elif name == "torch":
        import keras.src.backend.torch as module
    elif name == "numpy":
        if backend.backend() == "numpy":
            import keras.src.backend as module
        else:
            raise NotImplementedError(
                "Currently, we cannot dynamically import the numpy backend "
                "because it would disrupt the namespace of the import."
            )
    else:
        raise NotImplementedError
    return module
