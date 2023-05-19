import tensorflow as tf

from keras_aug import layers

num_classes = 10


class CutMixTest(tf.test.TestCase):
    def test_return_shapes(self):
        xs = tf.ones((2, 40, 40, 3))
        # randomly sample labels
        ys = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys = tf.squeeze(ys)
        ys = tf.one_hot(ys, num_classes)
        layer = layers.CutMix(seed=2023)
        outputs = layer({"images": xs, "labels": ys})

        xs, ys = outputs["images"], outputs["labels"]

        self.assertEqual(xs.shape, [2, 40, 40, 3])
        self.assertEqual(ys.shape, [2, 10])

    def test_cut_mix_call_results(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 3)), tf.ones((40, 40, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)
        layer = layers.CutMix(seed=2024)

        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # At least some pixels should be replaced in the CutMix operation
        self.assertTrue(tf.math.reduce_any(xs[0] == 1.0))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))
        # No labels should still be close to their original values
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_cut_mix_call_results_one_channel(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)
        layer = layers.CutMix(seed=2024)

        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # At least some pixels should be replaced in the CutMix operation
        self.assertTrue(tf.math.reduce_any(xs[0] == 1.0))
        self.assertTrue(tf.math.reduce_any(xs[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 1.0))
        self.assertTrue(tf.math.reduce_any(xs[1] == 2.0))
        # No labels should still be close to their original values
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_single_image_input(self):
        xs = tf.ones((4, 4, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = layers.CutMix()

        with self.assertRaisesRegexp(
            ValueError, "CutMix received a single image to `call`"
        ):
            _ = layer(inputs)

    def test_missing_labels(self):
        xs = tf.ones((2, 4, 4, 3))
        inputs = {"images": xs}
        layer = layers.CutMix()

        with self.assertRaisesRegexp(ValueError, "CutMix expects `labels`"):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 4, 4, 3))
        layer = layers.CutMix()

        with self.assertRaisesRegexp(
            ValueError,
            "CutMix expects `labels` or `segmentation_masks` to be present",
        ):
            _ = layer(xs)

    def test_cut_mix_call_segmentation_masks(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 3)), tf.ones((40, 40, 3))],
                axis=0,
            ),
            tf.float32,
        )
        masks = tf.cast(
            tf.stack(
                [2 * tf.ones((40, 40, 1)), tf.ones((40, 40, 1))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)
        layer = layers.CutMix(seed=2024)

        outputs = layer(
            {"images": xs, "labels": ys, "segmentation_masks": masks}
        )
        xs, ys = outputs["images"], outputs["labels"]
        masks = outputs["segmentation_masks"]

        # At least some pixels should be replaced in the CutMix operation
        self.assertTrue(tf.math.reduce_any(masks[0] == 1.0))
        self.assertTrue(tf.math.reduce_any(masks[0] == 2.0))
        self.assertTrue(tf.math.reduce_any(masks[1] == 1.0))
        self.assertTrue(tf.math.reduce_any(masks[1] == 2.0))
