import tensorflow as tf

from keras_aug import layers


class TrivialAugmentWideTest(tf.test.TestCase):
    def test_runs(self):
        trivial_augment = layers.TrivialAugmentWide(
            value_range=(0, 255),
            seed=2023,
        )
        xs = tf.ones((2, 4, 4, 3))

        ys = trivial_augment(xs)

        self.assertEqual(ys.shape, (2, 4, 4, 3))
