import tensorflow as tf

from keras_aug import augmentation


class RandomEraseTest(tf.test.TestCase):
    def _run_test(self, area_factor, aspect_ratio_factor):
        img_shape = (40, 40, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = augmentation.RandomErase(
            area_factor=area_factor,
            aspect_ratio_factor=aspect_ratio_factor,
            fill_mode="constant",
            fill_value=fill_value,
            seed=1,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_return_shapes_single_element(self):
        xs = tf.ones((40, 40, 3))

        layer = augmentation.RandomErase(
            area_factor=0.3, aspect_ratio_factor=1.0, seed=1
        )
        xs = layer(xs)

        self.assertEqual(xs.shape, [40, 40, 3])

    def test_random_cutout_single_float(self):
        self._run_test(0.5, 0.5)

    def test_random_cutout_tuple_float(self):
        self._run_test((0.4, 0.9), (0.1, 0.3))

    def test_random_cutout_fail_mix_bad_param_values(self):
        self.assertRaises(ValueError, lambda: self._run_test(0.5, (-1.0, 30)))

    def test_random_cutout_fail_reverse_lower_upper_float(self):
        self.assertRaises(ValueError, lambda: self._run_test(0.5, (0.9, 0.4)))

    def test_random_cutout_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))],
                axis=0,
            ),
            tf.float32,
        )

        patch_value = 0.0
        layer = augmentation.RandomErase(
            area_factor=0.5,
            aspect_ratio_factor=1.0,
            fill_mode="constant",
            fill_value=patch_value,
            seed=1,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == patch_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))

    def test_random_cutout_call_tiny_image(self):
        img_shape = (4, 4, 3)
        xs = tf.stack(
            [2 * tf.ones(img_shape), tf.ones(img_shape)],
            axis=0,
        )
        xs = tf.cast(xs, tf.float32)

        fill_value = 0.0
        layer = augmentation.RandomErase(
            area_factor=(0.4, 0.9),
            aspect_ratio_factor=(0.3, 1.0 / 0.3),
            fill_mode="constant",
            fill_value=fill_value,
            seed=1,
        )
        xs = layer(xs)

        # Some pixels should be replaced with fill value
        self.assertTrue(tf.math.reduce_any(xs[0] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == fill_value))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
