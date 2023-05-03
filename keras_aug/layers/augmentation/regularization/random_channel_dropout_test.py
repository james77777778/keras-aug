import tensorflow as tf

from keras_aug import layers


class RandomChannelDropoutTest(tf.test.TestCase):
    regular_args = {"factor": (0, 2)}

    def test_layer_created_with_invalid_factor(self):
        images = tf.ones(shape=(2, 32, 32, 3))
        args = self.regular_args.copy()
        args.update({"factor": (5, 5)})
        layer = layers.RandomChannelDropout(**args)

        with self.assertRaises(ValueError):
            _ = layer(images)

    def test_partially_zeros_out(self):
        images = tf.ones(shape=(2, 32, 32, 3))
        layer = layers.RandomChannelDropout(**self.regular_args)

        results = layer(images)

        self.assertEqual(tf.reduce_sum(results), 2 * 32 * 32 * 2)
