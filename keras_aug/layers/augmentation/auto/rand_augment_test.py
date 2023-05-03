import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers


class RandAugmentTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        ("0", 0),
        ("5", 5),
        ("10", 10),
        ("20", 20),
        ("30", 30),
    )
    def test_runs_with_magnitude(self, magnitude):
        rand_augment = layers.RandAugment(
            value_range=(0, 255),
            magnitude=magnitude,
            seed=2023,
        )
        xs = tf.ones((2, 4, 4, 3))

        ys = rand_augment(xs)

        self.assertEqual(ys.shape, (2, 4, 4, 3))
