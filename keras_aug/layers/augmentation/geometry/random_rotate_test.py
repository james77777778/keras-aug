import numpy as np
import tensorflow as tf
from keras_cv import bounding_box

from keras_aug import layers


class RandomRotationTest(tf.test.TestCase):
    regular_args = {
        "factor": 10,
        "fill_mode": "constant",
        "fill_value": 0,
        "interpolation": "bilinear",
        "bounding_box_format": "xyxy",
    }
    no_aug_args = {
        "factor": 0.0,
        "fill_mode": "constant",
        "fill_value": 0,
        "interpolation": "bilinear",
        "bounding_box_format": "xyxy",
    }

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = layers.RandomRotate(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.RandomRotate(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = layers.RandomRotate(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_rotate_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(np.float32)
        # 180 rotation
        args = self.no_aug_args.copy()
        args.update({"factor": (180, 180)})
        layer = layers.RandomRotate(**args)
        expected_output = np.asarray(
            [
                [24, 23, 22, 21, 20],
                [19, 18, 17, 16, 15],
                [14, 13, 12, 11, 10],
                [9, 8, 7, 6, 5],
                [4, 3, 2, 1, 0],
            ]
        ).astype(np.float32)

        output_image = layer(input_image)
        expected_output = np.reshape(expected_output, (5, 5, 1))

        self.assertAllClose(expected_output, output_image)

    def test_rotate_bounding_boxes(self):
        input_image = np.random.random((512, 512, 3)).astype(np.float32)
        bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[200, 200, 400, 400], [100, 100, 300, 300]], dtype=tf.float32
            ),
            "classes": tf.convert_to_tensor([1, 2], dtype=tf.float32),
        }
        input = {"images": input_image, "bounding_boxes": bounding_boxes}
        expected_bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [[112.0, 112.0, 312.0, 312.0], [212.0, 212.0, 412.0, 412.0]],
                dtype=tf.float32,
            ),
            "classes": tf.convert_to_tensor([1, 2], dtype=tf.float32),
        }
        # 180 rotation
        args = self.no_aug_args.copy()
        args.update({"factor": (180, 180)})
        layer = layers.RandomRotate(**args)

        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertAllClose(expected_bounding_boxes, output["bounding_boxes"])

    def test_rotate_ragged_bounding_boxes(self):
        input_image = np.random.random((2, 512, 512, 3)).astype(np.float32)
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[200, 200, 400, 400], [100, 100, 300, 300]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [[0, 0], [0]],
                dtype=tf.float32,
            ),
        }
        input = {"images": input_image, "bounding_boxes": bounding_boxes}
        expected_output = {
            "boxes": tf.ragged.constant(
                [
                    [
                        [112.0, 112.0, 312.0, 312.0],
                        [212.0, 212.0, 412.0, 412.0],
                    ],
                    [[112.0, 112.0, 312.0, 312.0]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant(
                [[0, 0], [0]],
                dtype=tf.float32,
            ),
        }
        # 180 rotation
        args = self.no_aug_args.copy()
        args.update({"factor": (180, 180)})
        layer = layers.RandomRotate(**args)

        output = layer(input)

        expected_output = bounding_box.to_dense(expected_output)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"],
            output["bounding_boxes"]["classes"],
        )

    def test_rotate_sparse_segmentation_mask(self):
        num_classes = 8

        input_images = np.random.random((2, 20, 20, 3)).astype(np.float32)
        # Masks are all 0s or 8s, to verify that when we rotate we don't do bad
        # mask interpolation to either a 0 or a 7
        masks = np.random.randint(2, size=(2, 20, 20, 1)) * (num_classes - 1)
        inputs = {"images": input_images, "segmentation_masks": masks}

        # 90 rotation
        args = self.no_aug_args.copy()
        args.update({"factor": (90, 90)})
        layer = layers.RandomRotate(**args)
        outputs = layer(inputs)
        expected_masks = np.rot90(masks, axes=(1, 2))
        self.assertAllClose(expected_masks, outputs["segmentation_masks"])

        # 45-degree rotation. Only verifies that no interpolation takes place.
        # 90 rotation
        args = self.no_aug_args.copy()
        args.update({"factor": (0.125, 0.125)})
        outputs = layer(inputs)
        self.assertAllInSet(outputs["segmentation_masks"], [0, 7])

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        args = self.no_aug_args.copy()
        args.update({"factor": (90, 90)})
        layer = layers.RandomRotate(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertAllInSet(result["segmentation_masks"], tf.range(0, 10))

    def test_ragged_segmentation_masks(self):
        images = tf.ragged.stack(
            [
                tf.random.uniform((8, 8, 3), dtype=tf.float32),
                tf.random.uniform((16, 8, 3), dtype=tf.float32),
            ]
        )
        segmentation_masks = tf.ragged.stack(
            [
                tf.random.uniform((8, 8, 1), maxval=10, dtype=tf.int32),
                tf.random.uniform((16, 8, 1), maxval=10, dtype=tf.int32),
            ]
        )
        segmentation_masks = tf.cast(segmentation_masks, dtype=tf.float32)
        args = self.no_aug_args.copy()
        args.update({"factor": (90, 90)})
        layer = layers.RandomRotate(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(
            isinstance(result["segmentation_masks"], tf.RaggedTensor)
        )
        self.assertAllInSet(
            result["segmentation_masks"].to_tensor(), tf.range(0, 10)
        )
