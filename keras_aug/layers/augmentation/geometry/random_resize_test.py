import numpy as np
import tensorflow as tf
from keras_cv import bounding_box

from keras_aug import layers


class RandomResizeTest(tf.test.TestCase):
    height, width = 224, 448
    regular_args = {
        "heights": [224],
        "interpolation": "bilinear",
        "antialias": False,
        "bounding_box_format": "rel_xyxy",
    }

    def test_no_adjustment(self):
        image_shape = (4, self.height, self.height, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = layers.RandomResize(**self.regular_args)
        output = layer(image)
        self.assertAllClose(image, output)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, self.height, self.height, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.RandomResize(**self.regular_args)
        output = layer(image)
        self.assertAllClose(image, output)

    def test_resize_image(self):
        input_image_shape = (4, self.height, self.width, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.RandomResize(**self.regular_args, seed=2023)

        output = layer(image)

        self.assertAllEqual(output.shape, (4, self.height, self.height, 3))

    def test_grayscale(self):
        input_image_shape = (4, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.RandomResize(**self.regular_args, seed=2023)

        output = layer(image)

        self.assertAllEqual(output.shape, (4, self.height, self.height, 1))

    def test_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, 300, 300, 3)
        mask_shape = (1, 300, 300, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.regular_args.copy()
        args.update({"heights": [150]})
        layer = layers.RandomResize(**args, seed=2023)
        input_mask_resized = tf.image.resize(mask, (150, 150), "nearest")

        output = layer(inputs)

        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.regular_args.copy()
        args.update({"heights": [233]})
        layer = layers.RandomResize(**args, seed=2023)

        output = layer(inputs)

        self.assertAllInSet(output["segmentation_masks"], [0, 7])

    def test_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]]),
            "classes": tf.convert_to_tensor([0]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.regular_args.copy()
        args.update({"heights": [10]})
        layer = layers.RandomResize(**args, seed=2023)
        expected_output = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }

        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertAllClose(
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
        args.update({"heights": [18]})
        layer = layers.RandomResize(**args, seed=2023)
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

        self.assertAllClose(
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
        args.update({"heights": [18]})
        layer = layers.RandomResize(**args, seed=2023)
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

        self.assertAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_same_shape_in_the_batch(self):
        images = tf.ragged.stack(
            [
                tf.ones((5, 5, 3)),
                tf.ones((8, 8, 3)),
            ]
        )
        args = self.regular_args.copy()
        args.update({"heights": [4, 6, 8, 10]})
        layer = layers.RandomResize(**args, seed=2023)

        for _ in range(3):
            output = layer(images)
            self.assertIn(output.shape[1], [4, 6, 8, 10])
            self.assertIn(output.shape[2], [4, 6, 8, 10])

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        args = self.regular_args.copy()
        args.update({"heights": [4, 6, 8, 10]})
        layer = layers.RandomResize(**args, seed=2023)

        for _ in range(3):
            output = layer(
                {"images": images, "segmentation_masks": segmentation_masks}
            )

            self.assertTrue(isinstance(output["segmentation_masks"], tf.Tensor))
            self.assertIn(output["segmentation_masks"].shape[1], [4, 6, 8, 10])
            self.assertIn(output["segmentation_masks"].shape[2], [4, 6, 8, 10])
            self.assertAllInSet(output["segmentation_masks"], tf.range(0, 10))

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
        args.update({"heights": [4, 6, 8, 10]})
        layer = layers.RandomResize(**args, seed=2023)

        for _ in range(3):
            output = layer(
                {"images": images, "segmentation_masks": segmentation_masks}
            )

            self.assertTrue(isinstance(output["segmentation_masks"], tf.Tensor))
            self.assertIn(output["segmentation_masks"].shape[1], [4, 6, 8, 10])
            self.assertIn(output["segmentation_masks"].shape[2], [4, 6, 8, 10])
            self.assertAllInSet(output["segmentation_masks"], tf.range(0, 10))
