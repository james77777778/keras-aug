import tensorflow as tf

from keras_aug import layers


class MosaicYOLOV8Test(tf.test.TestCase):
    num_classes = 10

    def test_return_shapes(self):
        xs = tf.ones((2, 32, 32, 3))
        # randomly sample labels
        ys_labels = tf.random.categorical(tf.math.log([[0.5, 0.5]]), 2)
        ys_labels = tf.squeeze(ys_labels)
        ys_labels = tf.one_hot(ys_labels, self.num_classes)

        # randomly sample bounding boxes
        ys_bounding_boxes = {
            "boxes": tf.random.uniform((2, 3, 4), 0, 1),
            "classes": tf.random.uniform((2, 3), 0, 1),
        }
        layer = layers.MosaicYOLOV8(
            height=32, width=32, bounding_box_format="xywh"
        )
        # augmentations.MosaicYOLOV8 on labels
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

        self.assertEqual(xs.shape, [2, 32, 32, 3])
        self.assertEqual(ys_labels.shape, [2, 10])
        self.assertEqual(ys_bounding_boxes["boxes"].shape, [2, None, 4])
        self.assertEqual(ys_bounding_boxes["classes"].shape, [2, None])

    def test_image_input_only(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((10, 10, 1)), tf.ones((10, 10, 1))], axis=0),
            tf.float32,
        )
        layer = layers.MosaicYOLOV8(height=32, width=32)
        with self.assertRaisesRegexp(
            ValueError, "expects inputs in a dictionary"
        ):
            _ = layer(xs)

    def test_single_image_input(self):
        xs = tf.ones((32, 32, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = layers.MosaicYOLOV8(height=32, width=32)
        with self.assertRaisesRegexp(
            ValueError,
            "MosaicYOLOV8 received a single image to `call`",
        ):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 32, 32, 3))
        layer = layers.MosaicYOLOV8(height=32, width=32)
        with self.assertRaisesRegexp(
            ValueError,
            "MosaicYOLOV8 expects inputs in a dictionary with format",
        ):
            _ = layer(xs)
