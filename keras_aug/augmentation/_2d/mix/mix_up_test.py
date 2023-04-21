import tensorflow as tf
from keras_cv import bounding_box

from keras_aug import augmentation


class MixUpTest(tf.test.TestCase):
    num_classes = 10

    def test_return_shapes(self):
        xs = tf.ones((2, 512, 512, 3))
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, self.num_classes)
        # randomly sample bounding boxes
        ys_bounding_boxes = {
            "boxes": tf.random.uniform((2, 3, 4), 0, 1),
            "classes": tf.random.uniform((2, 3), 0, 1),
        }
        layer = augmentation.MixUp()

        # mixup on labels
        outputs = layer(
            {
                "images": xs,
                "labels": ys_labels,
                "bounding_boxes": ys_bounding_boxes,
            }
        )
        xs, ys_labels, ys_bounding_boxes = (
            outputs["images"],
            outputs["labels"],
            outputs["bounding_boxes"],
        )
        ys_bounding_boxes = bounding_box.to_dense(ys_bounding_boxes)

        self.assertEqual(xs.shape, [2, 512, 512, 3])
        self.assertEqual(ys_labels.shape, [2, 10])
        self.assertEqual(ys_bounding_boxes["boxes"].shape, [2, 6, 4])
        self.assertEqual(ys_bounding_boxes["classes"].shape, [2, 6])

    def test_mix_up_call_results(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = augmentation.MixUp()
        outputs = layer({"images": xs, "labels": ys})
        xs, ys = outputs["images"], outputs["labels"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_in_tf_function(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((4, 4, 3)), tf.ones((4, 4, 3))],
                axis=0,
            ),
            tf.float32,
        )
        ys = tf.one_hot(tf.constant([0, 1]), 2)

        layer = augmentation.MixUp()

        @tf.function
        def augment(x, y):
            return layer({"images": x, "labels": y})

        outputs = augment(xs, ys)
        xs, ys = outputs["images"], outputs["labels"]

        # None of the individual values should still be close to 1 or 0
        self.assertNotAllClose(xs, 1.0)
        self.assertNotAllClose(xs, 2.0)

        # No labels should still be close to their originals
        self.assertNotAllClose(ys, 1.0)
        self.assertNotAllClose(ys, 0.0)

    def test_image_input_only(self):
        xs = tf.cast(
            tf.stack(
                [2 * tf.ones((100, 100, 1)), tf.ones((100, 100, 1))], axis=0
            ),
            tf.float32,
        )
        layer = augmentation.MixUp()
        with self.assertRaisesRegexp(
            ValueError, "expects inputs in a dictionary"
        ):
            _ = layer(xs)

    def test_single_image_input(self):
        xs = tf.ones((512, 512, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = augmentation.MixUp()
        with self.assertRaisesRegexp(
            ValueError, "MixUp received a single image to `call`"
        ):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 512, 512, 3))
        layer = augmentation.MixUp()
        with self.assertRaisesRegexp(
            ValueError, "MixUp expects inputs in a dictionary with format"
        ):
            _ = layer(xs)
