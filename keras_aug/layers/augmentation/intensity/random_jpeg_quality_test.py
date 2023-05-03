import tensorflow as tf
from absl.testing import parameterized

from keras_aug import layers


class RandomJpegQualityTest(tf.test.TestCase, parameterized.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "factor": (75, 100),  # 25%
    }

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0
        args = self.regular_args.copy()
        args.update({"value_range": (0, 100)})
        layer = layers.RandomJpegQuality(**args)

        output = layer(image)

        self.assertNotAllClose(image, output)

    def test_in_single_image(self):
        layer = layers.RandomJpegQuality(**self.regular_args)
        # RGB
        xs = tf.cast(
            tf.ones((512, 512, 3)) * 255.0,
            dtype=tf.float32,
        )

        xs = layer(xs)

        self.assertEqual(xs.shape, [512, 512, 3])

        # greyscale
        xs = tf.cast(
            tf.ones((512, 512, 1)) * 255.0,
            dtype=tf.float32,
        )

        xs = layer(xs)

        self.assertEqual(xs.shape, [512, 512, 1])
