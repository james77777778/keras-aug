"""
Most of these codes come from KerasCV.
"""
import tensorflow as tf

from keras_aug.datapoints import bounding_box


class ToDenseTest(tf.test.TestCase):
    def test_converts_to_dense(self):
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]
            ),
            "classes": tf.ragged.constant([[0], [1, 2, 3]]),
        }

        bounding_boxes = bounding_box.to_dense(bounding_boxes)

        self.assertEqual(bounding_boxes["boxes"].shape, [2, 3, 4])
        self.assertEqual(bounding_boxes["classes"].shape, [2, 3])

    def test_converts_to_dense_with_max_boxes(self):
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]]
            ),
            "classes": tf.ragged.constant([[0], [1, 2, 3]]),
        }

        bounding_boxes = bounding_box.to_dense(bounding_boxes, max_boxes=16)

        self.assertEqual(bounding_boxes["boxes"].shape, [2, 16, 4])
        self.assertEqual(bounding_boxes["classes"].shape, [2, 16])
