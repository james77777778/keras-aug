import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras_cv import bounding_box

from keras_aug import layers


class RandomFlipTest(tf.test.TestCase, parameterized.TestCase):
    def test_horizontal_flip(self):
        np.random.seed(1337)
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        inp = np.random.random((2, 5, 8, 3))
        expected_output = np.flip(inp, axis=2)
        layer = layers.RandomFlip("horizontal")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(inp)
            self.assertAllClose(expected_output, actual_output)

    def test_flip_ragged(self):
        images = tf.ragged.stack([tf.ones((4, 4, 3)), tf.ones((4, 8, 3))])
        bounding_boxes = {
            "boxes": tf.ragged.stack([tf.ones((5, 4)), tf.ones((3, 4))]),
            "classes": tf.ragged.stack([tf.ones((5,)), tf.ones((3,))]),
        }
        inputs = {"images": images, "bounding_boxes": bounding_boxes}
        layer = layers.RandomFlip(mode="horizontal", bounding_box_format="xywh")
        _ = layer(inputs)

    def test_vertical_flip(self):
        np.random.seed(1337)
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        inp = np.random.random((2, 5, 8, 3))
        expected_output = np.flip(inp, axis=1)
        layer = layers.RandomFlip("vertical")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(inp)
            self.assertAllClose(expected_output, actual_output)

    def test_flip_both(self):
        np.random.seed(1337)
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        inp = np.random.random((2, 5, 8, 3))
        expected_output = np.flip(inp, axis=2)
        expected_output = np.flip(expected_output, axis=1)
        layer = layers.RandomFlip("horizontal_and_vertical")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(inp)
        self.assertAllClose(expected_output, actual_output)

    def test_random_flip_default(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = np.flip(input_images, axis=2)
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        layer = layers.RandomFlip()
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(input_images)
            self.assertAllClose(expected_output, actual_output)

    def test_random_flip_low_rate(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = input_images
        # mock_random > 0.5 but no flipping occurs due to low rate
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        layer = layers.RandomFlip(rate=0.1)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(input_images)
        self.assertAllClose(expected_output, actual_output)

    def test_random_flip_high_rate(self):
        input_images = np.random.random((2, 5, 8, 3)).astype(np.float32)
        expected_output = np.flip(input_images, axis=2)
        # mock_random is small (0.2) but flipping still occurs due to high rate
        mock_random = tf.convert_to_tensor([[0.2], [0.2]])
        layer = layers.RandomFlip(rate=0.9)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(input_images)
        self.assertAllClose(expected_output, actual_output)

    def test_config_with_custom_name(self):
        layer = layers.RandomFlip(name="image_preproc")
        config = layer.get_config()
        layer_1 = layers.RandomFlip.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_random_flip_unbatched_image(self):
        input_image = np.random.random((4, 4, 1)).astype(np.float32)
        expected_output = np.flip(input_image, axis=0)
        mock_random = tf.convert_to_tensor([[0.6]])
        layer = layers.RandomFlip("vertical")
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            actual_output = layer(input_image)
            self.assertAllClose(expected_output, actual_output)

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = layers.RandomFlip()
        self.assertAllEqual(layer(inputs).dtype, "float32")
        layer = layers.RandomFlip(dtype="uint8")
        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_augment_bounding_box_batched_input(self):
        image = tf.zeros([20, 20, 3])
        bounding_boxes = {
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 10, 10], [4, 4, 12, 12]],
                    [[4, 4, 12, 12], [0, 0, 10, 10]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.convert_to_tensor(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        }

        input = {"images": [image, image], "bounding_boxes": bounding_boxes}
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        layer = layers.RandomFlip(
            "horizontal_and_vertical", bounding_box_format="xyxy"
        )
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            output = layer(input)

        expected_output = {
            "boxes": tf.convert_to_tensor(
                [
                    [[10, 10, 20, 20], [8, 8, 16, 16]],
                    [[8, 8, 16, 16], [10, 10, 20, 20]],
                ]
            ),
            "classes": tf.convert_to_tensor(
                [
                    [0, 0],
                    [0, 0],
                ]
            ),
        }
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
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 10, 10], [4, 4, 12, 12]], [[0, 0, 10, 10]]],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 0], [0]], dtype=tf.float32),
        }

        input = {"images": image, "bounding_boxes": bounding_boxes}
        mock_random = tf.convert_to_tensor([[0.6], [0.6]])
        layer = layers.RandomFlip(
            "horizontal_and_vertical", bounding_box_format="xyxy"
        )
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            output = layer(input)

        expected_output = {
            "boxes": tf.ragged.constant(
                [[[10, 10, 20, 20], [8, 8, 16, 16]], [[10, 10, 20, 20]]],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 0], [0]], dtype=tf.float32),
        }

        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )
        expected_output = bounding_box.to_dense(expected_output)
        self.assertAllClose(
            expected_output["boxes"], output["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_augment_segmentation_mask(self):
        np.random.seed(1337)
        image = np.random.random((1, 20, 20, 3)).astype(np.float32)
        mask = np.random.randint(2, size=(1, 20, 20, 1)).astype(np.float32)

        input = {"images": image, "segmentation_masks": mask}

        # Flip both vertically and horizontally
        mock_random = tf.convert_to_tensor([[0.6]])
        layer = layers.RandomFlip("horizontal_and_vertical")

        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_random,
        ):
            output = layer(input)

        expected_mask = np.flip(np.flip(mask, axis=1), axis=2)

        self.assertAllClose(expected_mask, output["segmentation_masks"])

    def test_ragged_bounding_boxes(self):
        input_image = np.random.random((2, 512, 512, 3)).astype(np.float32)
        bounding_boxes = {
            "boxes": tf.ragged.constant(
                [
                    [[200, 200, 400, 400], [100, 100, 300, 300]],
                    [[200, 200, 400, 400]],
                ],
                dtype=tf.float32,
            ),
            "classes": tf.ragged.constant([[0, 0], [0]], dtype=tf.float32),
        }

        input = {"images": input_image, "bounding_boxes": bounding_boxes}
        layer = layers.RandomFlip(bounding_box_format="xyxy")
        _ = layer(input)

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        layer = layers.RandomFlip()

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
        layer = layers.RandomFlip()

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(
            isinstance(result["segmentation_masks"], tf.RaggedTensor)
        )
        self.assertAllInSet(
            result["segmentation_masks"].to_tensor(0), tf.range(0, 10)
        )
