import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras_cv import bounding_box

from keras_aug import layers


class PadIfNeededTest(tf.test.TestCase, parameterized.TestCase):
    height, width = 32, 32
    regular_args = {
        "min_height": 32,
        "min_width": 32,
        "pad_height_divisor": None,
        "pad_width_divisor": None,
        "position": "center",
        "padding_value": 0,
        "bounding_box_format": "rel_xyxy",
    }

    def test_augments_image(self):
        input_image_shape = (4, 50, 60, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.PadIfNeeded(**self.regular_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertNotAllClose(output, input_image_resized)

    def test_grayscale(self):
        input_image_shape = (4, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.PadIfNeeded(**self.regular_args, seed=2023)

        output = layer(image)

        self.assertAllEqual(output.shape, (4, self.height, self.width, 1))

    def test_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, 50, 50, 3)
        mask_shape = (1, 50, 50, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.regular_args.copy()
        args.update({"min_height": 100, "min_width": 100})
        layer = layers.PadIfNeeded(**args, seed=2023)

        output = layer(inputs)

        self.assertAllClose(
            output["segmentation_masks"][:, 25:-25, 25:-25, :], mask
        )

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.regular_args.copy()
        args.update({"min_height": 123, "min_width": 123})
        layer = layers.PadIfNeeded(**args, seed=2023)

        output = layer(inputs)

        self.assertAllInSet(output["segmentation_masks"], [0, num_classes - 1])

    def test_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]]),
            "classes": tf.convert_to_tensor([0]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.regular_args.copy()
        args.update({"min_height": 30, "min_width": 30})
        layer = layers.PadIfNeeded(**args, seed=2023)
        expected_output = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }

        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertNotAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_augment_boxes_batched_input(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ]
            ),
            "classes": tf.convert_to_tensor([[0, 0], [0, 0]]),
        }
        input = {"images": [image, image], "bounding_boxes": boxes}
        args = self.regular_args.copy()
        args.update({"min_height": 30, "min_width": 30})
        layer = layers.PadIfNeeded(**args, seed=2023)
        expected_output = {
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ]
            ),
            "classes": tf.convert_to_tensor([[0, 0], [0, 0]]),
        }

        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertNotAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant([[0, 0], [0]]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.regular_args.copy()
        args.update({"min_height": 30, "min_width": 30})
        layer = layers.PadIfNeeded(**args, seed=2023)
        # the result boxes will still have the entire image in them
        expected_output = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant([[0, 0], [0]]),
        }
        expected_output = bounding_box.to_dense(expected_output)

        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertNotAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_padding_center(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "center"})
        layer = layers.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, 0:2, :, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, -2:, :, :]), 0.0)

    def test_padding_top(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "top_left"})
        layer = layers.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertNotEqual(tf.reduce_mean(outputs[:, 0:2, :, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, -4:, :, :]), 0.0)

    def test_padding_bottom(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.regular_args.copy()
        args.update(
            {"min_height": 8, "min_width": 8, "position": "bottom_left"}
        )
        layer = layers.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, 0:4, :, :]), 0.0)
        self.assertNotEqual(tf.reduce_mean(outputs[:, -2:, :, :]), 0.0)

    def test_padding_left(self):
        inputs = tf.ones((1, 8, 4, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "top_left"})
        layer = layers.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertNotEqual(tf.reduce_mean(outputs[:, :, 0:2, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, :, -2:, :]), 0.0)

    def test_padding_right(self):
        inputs = tf.ones((1, 8, 4, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "top_right"})
        layer = layers.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, :, 0:4, :]), 0.0)
        self.assertNotEqual(tf.reduce_mean(outputs[:, :, -2:, :]), 0.0)

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        args = self.regular_args.copy()
        args.update(
            {"min_height": 16, "min_width": 16, "position": "top_right"}
        )
        layer = layers.PadIfNeeded(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (16, 16))
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
        args = self.regular_args.copy()
        args.update(
            {"min_height": 16, "min_width": 16, "position": "top_right"}
        )
        layer = layers.PadIfNeeded(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(
            isinstance(result["segmentation_masks"], tf.RaggedTensor)
        )
        self.assertAllInSet(
            result["segmentation_masks"].to_tensor(), tf.range(0, 10)
        )
