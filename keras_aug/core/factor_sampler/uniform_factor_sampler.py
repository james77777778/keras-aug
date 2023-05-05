import keras
import keras.backend

from keras_aug.core.factor_sampler.factor_sampler import FactorSampler


@keras.utils.register_keras_serializable(package="keras_aug")
class UniformFactorSampler(FactorSampler):
    """UniformFactorSampler samples factors uniformly from a range.

    Args:
        lower (float): the lower bound of values returned from ``__call__()``.
        upper (float): the upper bound of values returned from ``__call__()``.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
    """

    def __init__(self, lower, upper, seed=None):
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.rng = keras.backend.RandomGenerator(
            seed=seed, rng_type="stateless"
        )

    def __call__(self, shape=(), dtype="float32"):
        return self.rng.random_uniform(
            shape,
            minval=self.lower,
            maxval=self.upper,
            dtype=dtype,
        )

    def get_config(self):
        return {
            "lower": self.lower,
            "upper": self.upper,
            "seed": self.seed,
        }
