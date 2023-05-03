import tensorflow as tf

from keras_aug import layers


class RandomSharpnessTest(tf.test.TestCase):
    def test_random_sharpness_blur_effect_single_channel(self):
        xs = tf.expand_dims(
            tf.constant(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            axis=-1,
        )
        xs = tf.expand_dims(xs, axis=0)
        layer = layers.RandomSharpness(value_range=(0, 255), factor=(0.0, 0.0))
        result = tf.expand_dims(
            tf.constant(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1 / 13, 1 / 13, 1 / 13, 0, 0],
                    [0, 0, 1 / 13, 5 / 13, 1 / 13, 0, 0],
                    [0, 0, 1 / 13, 1 / 13, 1 / 13, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ]
            ),
            axis=-1,
        )
        result = tf.expand_dims(result, axis=0)

        ys = layer(xs)

        self.assertEqual(xs.shape, ys.shape)
        self.assertAllClose(ys, result)
