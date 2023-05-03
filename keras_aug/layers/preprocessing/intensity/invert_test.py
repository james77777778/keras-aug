import tensorflow as tf

from keras_aug import layers


class InvertTest(tf.test.TestCase):
    regular_args = {
        "value_range": (0, 255),
    }

    def test_invert_output(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        expected_output = 255.0 - image

        layer = layers.Invert(**self.regular_args)
        output = layer(image)

        self.assertAllClose(output, expected_output)
