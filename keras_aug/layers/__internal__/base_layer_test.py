import tempfile

import tensorflow as tf
from keras import backend

from keras_aug.layers.__internal__.base_layer import BaseRandomLayer


class BaseRandomLayerTest(tf.test.TestCase):
    def test_saved_model(self):
        base_random_layer = BaseRandomLayer(
            force_generator=True, rng_type="stateless"
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            tf.saved_model.save(base_random_layer, tmpdirname)
            tf.saved_model.load(tmpdirname)

    def test_checkpoint(self):
        inputs = tf.keras.Input(shape=(1,))
        base_random_layer = BaseRandomLayer(
            force_generator=True, rng_type="stateless"
        )
        outputs = base_random_layer(inputs)
        model = tf.keras.Model(inputs, outputs)
        checkpoint = tf.train.Checkpoint(model)

        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = checkpoint.save(tmpdirname)
            checkpoint.restore(save_path)

    def test_lookup_dependency(self):
        base_random_layer = BaseRandomLayer(
            force_generator=True, rng_type="stateless"
        )

        rng = base_random_layer._lookup_dependency("_random_generator")

        self.assertTrue(isinstance(rng, backend.RandomGenerator))
