import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_aug import layers


class AddOneToInputs(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_counter = tf.Variable(initial_value=0)

    def call(self, inputs):
        result = inputs.copy()
        result["images"] = inputs["images"] + 1
        self.call_counter.assign_add(1)
        return result


class RandomAugmentationPipelineTest(tf.test.TestCase, parameterized.TestCase):
    def test_calls_layer_augmentation_per_image(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])
        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_calls_layer_augmentation_in_graph(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])

        @tf.function()
        def call_pipeline(xs):
            return pipeline(xs)

        xs = tf.random.uniform((2, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = call_pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_batchwise(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer], batchwise=True)
        xs = tf.random.uniform((4, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)
        # Ensure the layer is only called once for the entire batch
        self.assertEqual(layer.call_counter, 1)

    def test_works_with_cutmix_mixup(self):
        pipeline = layers.RandomChoice(
            layers=[layers.CutMix(), layers.MixUp()], batchwise=True
        )
        xs = {
            "images": tf.random.uniform((4, 5, 5, 3), 0, 100, dtype=tf.float32),
            "labels": tf.random.uniform((4, 10), 0, 1, dtype=tf.float32),
        }
        pipeline(xs)

    def test_calls_layer_augmentation_single_image(self):
        layer = AddOneToInputs()
        pipeline = layers.RandomChoice(layers=[layer])
        xs = tf.random.uniform((5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

    def test_calls_choose_one_layer_augmentation(self):
        batch_size = 10
        pipeline = layers.RandomChoice(
            layers=[AddOneToInputs(), AddOneToInputs()]
        )
        xs = tf.random.uniform((batch_size, 5, 5, 3), 0, 100, dtype=tf.float32)
        os = pipeline(xs)

        self.assertAllClose(xs + 1, os)

        total_calls = (
            pipeline.layers[0].call_counter + pipeline.layers[1].call_counter
        )
        self.assertEqual(total_calls, batch_size)
