import tensorflow as tf

from keras_aug import augmentation


class RescalingTest(tf.test.TestCase):
    regular_args = {
        "scale": 1.0 / 255.0,
        "offset": 0.0,
    }
    no_aug_args = {
        "scale": 1.0,
        "offset": 0.0,
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape)

        layer = augmentation.Rescaling(**self.no_aug_args)
        output = layer(image)

        self.assertAllClose(image, output)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = augmentation.Rescaling(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_rescaling_output(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        scale = tf.convert_to_tensor(self.regular_args["scale"])
        scale = tf.reshape(scale, shape=(1, 1, 1, 1))
        offset = tf.convert_to_tensor(self.regular_args["offset"])
        offset = tf.reshape(offset, shape=(1, 1, 1, 1))
        expected_output = image * scale + offset

        layer = augmentation.Rescaling(**self.regular_args)
        output = layer(image)

        self.assertAllClose(output, expected_output)
