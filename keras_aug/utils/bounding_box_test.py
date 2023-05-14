import tensorflow as tf

from keras_aug.utils import bounding_box as bounding_box_utils


class BoundingBoxUtilsTest(tf.test.TestCase):
    def test_sanitize_bounding_boxes_intact(self):
        bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[[10, 10, 20, 20], [10, 10, 30, 30]]], dtype=tf.float32
            ),
            "classes": tf.convert_to_tensor([[1, 2]], dtype=tf.float32),
        }
        expected_classes = tf.convert_to_tensor([[1, 2]], dtype=tf.float32)

        result_bounding_boxes = bounding_box_utils.sanitize_bounding_boxes(
            bounding_boxes
        )

        self.assertAllEqual(result_bounding_boxes["classes"], expected_classes)

    def test_sanitize_bounding_boxes_by_min_size(self):
        images = tf.ones((1, 50, 50, 3))
        bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[[10, 10, 20, 20], [10, 10, 30, 30]]], dtype=tf.float32
            ),
            "classes": tf.convert_to_tensor([[1, 2]], dtype=tf.float32),
        }
        expected_classes = tf.convert_to_tensor([[-1, 2]], dtype=tf.float32)

        result_bounding_boxes = bounding_box_utils.sanitize_bounding_boxes(
            bounding_boxes,
            min_size=25,
            bounding_box_format="xywh",
            images=images,
        )

        self.assertAllEqual(result_bounding_boxes["classes"], expected_classes)

    def test_sanitize_bounding_boxes_by_min_area_ratio(self):
        images = tf.ones((1, 50, 50, 3))
        ref_images = tf.ones((1, 50, 50, 3))
        bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[[10, 10, 20, 20], [10, 10, 30, 30]]], dtype=tf.float32
            ),
            "classes": tf.convert_to_tensor([[1, 2]], dtype=tf.float32),
        }
        ref_bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[[10, 10, 25, 25], [10, 10, 100, 100]]], dtype=tf.float32
            ),
            "classes": tf.convert_to_tensor([[1, 2]], dtype=tf.float32),
        }
        expected_classes = tf.convert_to_tensor([[1, -1]], dtype=tf.float32)

        result_bounding_boxes = bounding_box_utils.sanitize_bounding_boxes(
            bounding_boxes,
            min_area_ratio=0.1,
            bounding_box_format="xywh",
            reference_bounding_boxes=ref_bounding_boxes,
            images=images,
            reference_images=ref_images,
        )

        self.assertAllEqual(result_bounding_boxes["classes"], expected_classes)

    def test_sanitize_bounding_boxes_by_max_aspect_ratio(self):
        images = tf.ones((1, 50, 50, 3))
        bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[[10, 10, 20, 20], [10, 10, 30, 3001]]], dtype=tf.float32
            ),
            "classes": tf.convert_to_tensor([[1, 2]], dtype=tf.float32),
        }
        expected_classes = tf.convert_to_tensor([[1, -1]], dtype=tf.float32)

        result_bounding_boxes = bounding_box_utils.sanitize_bounding_boxes(
            bounding_boxes,
            max_aspect_ratio=100.0,
            bounding_box_format="xywh",
            images=images,
        )

        self.assertAllEqual(result_bounding_boxes["classes"], expected_classes)
