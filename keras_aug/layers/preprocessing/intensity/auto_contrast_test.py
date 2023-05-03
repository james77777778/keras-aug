import tensorflow as tf

from keras_aug import layers


class AutoContrastTest(tf.test.TestCase):
    def test_constant_channels_dont_get_nanned(self):
        img = tf.constant([1, 1], dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 1.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 1.0))

    def test_auto_contrast_expands_value_range(self):
        img = tf.constant([0, 128], dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 255.0))

    def test_auto_contrast_different_values_per_channel(self):
        img = tf.constant(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=tf.float32,
        )
        img = tf.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0, ..., 0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0, ..., 1] == 0.0))

        self.assertTrue(tf.math.reduce_any(ys[0, ..., 0] == 255.0))
        self.assertTrue(tf.math.reduce_any(ys[0, ..., 1] == 255.0))

        self.assertAllClose(
            ys,
            [
                [
                    [[0.0, 0.0, 0.0], [85.0, 85.0, 85.0]],
                    [[170.0, 170.0, 170.0], [255.0, 255.0, 255.0]],
                ]
            ],
        )

    def test_auto_contrast_expands_value_range_uint8(self):
        img = tf.constant([0, 128], dtype=tf.uint8)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 255))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 255.0))

    def test_auto_contrast_properly_converts_value_range(self):
        img = tf.constant([0, 0.5], dtype=tf.float32)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=-1)
        img = tf.expand_dims(img, axis=0)

        layer = layers.AutoContrast(value_range=(0, 1))
        ys = layer(img)

        self.assertTrue(tf.math.reduce_any(ys[0] == 0.0))
        self.assertTrue(tf.math.reduce_any(ys[0] == 1.0))
