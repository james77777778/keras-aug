import tensorflow as tf

from keras_aug import layers


class IdentityTest(tf.test.TestCase):
    def test_all_inputs_intact(self):
        images = tf.random.uniform(shape=(2, 4, 4, 3))
        labels = tf.random.uniform(
            shape=(2, 1), minval=0, maxval=10, dtype=tf.int32
        )
        bounding_boxes = {
            "boxes": tf.ragged.stack(
                [
                    tf.ones((2, 4)),
                    tf.ones((3, 4)),
                ]
            ),
            "classes": tf.ragged.stack(
                [
                    tf.ones((2,)),
                    tf.ones((3,)),
                ]
            ),
        }
        segmentation_masks = tf.random.uniform(
            shape=(2, 4, 4, 3), minval=0, maxval=10, dtype=tf.int32
        )
        inputs = {
            "images": images,
            "labels": labels,
            "bounding_boxes": bounding_boxes,
            "segmentation_masks": segmentation_masks,
        }
        layer = layers.Identity()

        outputs = layer(inputs)

        for key in inputs:
            if key == "bounding_boxes":
                self.assertAllEqual(outputs[key]["boxes"], inputs[key]["boxes"])
                self.assertAllEqual(
                    outputs[key]["classes"], inputs[key]["classes"]
                )
            else:
                self.assertAllEqual(outputs[key], inputs[key])
