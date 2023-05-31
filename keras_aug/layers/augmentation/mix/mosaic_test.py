import tensorflow as tf

from keras_aug import layers


class MosaicTest(tf.test.TestCase):
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
        layer = layers.Mosaic(height=32, width=32, bounding_box_format="xywh")
        # mosaic on labels
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
        self.assertEqual(ys_bounding_boxes["boxes"].shape, [2, 3 * 4, 4])
        self.assertEqual(ys_bounding_boxes["classes"].shape, [2, 3 * 4])

    def test_image_input_only(self):
        xs = tf.cast(
            tf.stack([2 * tf.ones((10, 10, 1)), tf.ones((10, 10, 1))], axis=0),
            tf.float32,
        )
        layer = layers.Mosaic(height=32, width=32)
        with self.assertRaisesRegexp(
            ValueError, "expects inputs in a dictionary"
        ):
            _ = layer(xs)

    def test_single_image_input(self):
        xs = tf.ones((32, 32, 3))
        ys = tf.one_hot(tf.constant([1]), 2)
        inputs = {"images": xs, "labels": ys}
        layer = layers.Mosaic(height=32, width=32)
        with self.assertRaisesRegexp(
            ValueError,
            "Mosaic received a single image to `call`",
        ):
            _ = layer(inputs)

    def test_image_input(self):
        xs = tf.ones((2, 32, 32, 3))
        layer = layers.Mosaic(height=32, width=32)
        with self.assertRaisesRegexp(
            ValueError,
            "Mosaic expects inputs in a dictionary with format",
        ):
            _ = layer(xs)

    def test_ragged_input_with_graph_mode(self):
        images = tf.ragged.stack(
            [
                tf.random.uniform((8, 8, 3), dtype=tf.float32),
                tf.random.uniform((16, 8, 3), dtype=tf.float32),
                tf.random.uniform((8, 8, 3), dtype=tf.float32),
                tf.random.uniform((16, 8, 3), dtype=tf.float32),
            ]
        )
        # randomly sample labels
        labels = tf.random.categorical(tf.math.log([[0.5, 0.5, 0.5, 0.5]]), 2)
        labels = tf.squeeze(labels)
        labels = tf.one_hot(labels, 10)
        segmentation_masks = tf.ragged.stack(
            [
                tf.random.uniform((8, 8, 1), maxval=10, dtype=tf.int32),
                tf.random.uniform((16, 8, 1), maxval=10, dtype=tf.int32),
                tf.random.uniform((8, 8, 1), maxval=10, dtype=tf.int32),
                tf.random.uniform((16, 8, 1), maxval=10, dtype=tf.int32),
            ]
        )
        layer = layers.Mosaic(height=8, width=8)

        @tf.function
        def fn(inputs):
            outputs = layer(inputs)
            image_shape = outputs["images"].shape
            segmentation_mask_shape = outputs["segmentation_masks"].shape
            self.assertEqual(image_shape, (4, 8, 8, 3))
            self.assertEqual(segmentation_mask_shape, (4, 8, 8, 1))

        fn(
            {
                "images": images,
                "labels": labels,
                "segmentation_masks": segmentation_masks,
            }
        )
