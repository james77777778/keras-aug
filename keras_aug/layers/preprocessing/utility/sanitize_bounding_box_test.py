import tensorflow as tf

from keras_aug import layers
from keras_aug.utils.augmentation import BOUNDING_BOXES
from keras_aug.utils.augmentation import IMAGES


class SanitizeBoundingBoxTest(tf.test.TestCase):
    def test_sanitize_small_bounding_boxes(self):
        images = tf.random.uniform((2, 512, 512, 3))
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[10, 10, 20, 20], [100, 100, 150, 150]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 1], [2]], dtype=tf.float32),
        }
        expected_classes = tf.ragged.constant([[1], [2]], dtype=tf.float32)
        layer = layers.SanitizeBoundingBox(
            min_size=35, bounding_box_format="xyxy"
        )

        output = layer({IMAGES: images, BOUNDING_BOXES: bounding_boxes})

        self.assertAllEqual(output[BOUNDING_BOXES]["classes"], expected_classes)
