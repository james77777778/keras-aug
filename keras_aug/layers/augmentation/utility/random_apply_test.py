import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_aug import layers
from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)


class ZeroOut(VectorizedBaseRandomLayer):
    """Zero out all entries, for testing purposes."""

    def __init__(self):
        super(ZeroOut, self).__init__()

    def augment_images(self, image, transformation=None, **kwargs):
        return 0 * image

    def augment_labels(self, label, transformation=None, **kwargs):
        return 0 * label


class RandomApplyTest(tf.test.TestCase, parameterized.TestCase):
    rng = tf.random.Generator.from_seed(seed=1234)

    @parameterized.parameters([-0.5, 1.7])
    def test_raises_error_on_invalid_rate_parameter(self, invalid_rate):
        with self.assertRaises(ValueError):
            layers.RandomApply(rate=invalid_rate, layer=ZeroOut())

    def test_inputs_unchanged_with_zero_rate(self):
        dummy_inputs = self.rng.uniform(shape=(4, 16, 16, 3))
        layer = layers.RandomApply(rate=0.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)

        self.assertAllClose(outputs, dummy_inputs)

    def test_all_inputs_changed_with_rate_equal_to_one(self):
        dummy_inputs = self.rng.uniform(shape=(4, 16, 16, 3))
        layer = layers.RandomApply(rate=1.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)

        self.assertAllEqual(outputs, tf.zeros_like(dummy_inputs))

    def test_works_with_single_image(self):
        dummy_inputs = self.rng.uniform(shape=(16, 16, 3))
        layer = layers.RandomApply(rate=1.0, layer=ZeroOut())

        outputs = layer(dummy_inputs)

        self.assertAllEqual(outputs, tf.zeros_like(dummy_inputs))

    def test_can_modify_label(self):
        dummy_inputs = self.rng.uniform(shape=(4, 16, 16, 3))
        dummy_labels = tf.ones(shape=(4, 2))
        layer = layers.RandomApply(rate=1.0, layer=ZeroOut())

        outputs = layer({"images": dummy_inputs, "labels": dummy_labels})

        self.assertAllEqual(outputs["labels"], tf.zeros_like(dummy_labels))

    def test_works_with_native_keras_layers(self):
        dummy_inputs = self.rng.uniform(shape=(4, 16, 16, 3))
        zero_out = keras.layers.Lambda(lambda x: {"images": 0 * x["images"]})
        layer = layers.RandomApply(rate=1.0, layer=zero_out)

        outputs = layer(dummy_inputs)

        self.assertAllEqual(outputs, tf.zeros_like(dummy_inputs))

    def test_works_with_xla(self):
        dummy_inputs = self.rng.uniform(shape=(4, 16, 16, 3))
        # auto_vectorize=True will crash XLA
        layer = layers.RandomApply(rate=0.5, layer=ZeroOut())

        @tf.function(jit_compile=True)
        def apply(x):
            return layer(x)

        apply(dummy_inputs)
