import unittest

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import keras

from keras_aug import layers


class RandomCropTest(tf.test.TestCase, parameterized.TestCase):
    def test_input_smaller_than_crop_box(self):
        np.random.seed(1337)
        height, width = 10, 8
        inp = np.random.random((12, 3, 3, 3))
        layer = layers.RandomCrop(height, width)
        actual_output = layer(inp)
        # In this case, output should equal resizing with crop_to_aspect
        # ratio.
        resizing_layer = keras.layers.Resizing(height, width)
        expected_output = resizing_layer(inp)
        self.assertAllEqual(expected_output, actual_output)

    def test_training_with_mock(self):
        np.random.seed(1337)
        batch_size = 12
        height, width = 3, 4
        height_offset = np.random.randint(low=0, high=3)
        width_offset = np.random.randint(low=0, high=5)
        # manually compute transformations which shift height_offset and
        # width_offset respectively
        tops = tf.ones((batch_size, 1)) * (height_offset / (5 - height))
        lefts = tf.ones((batch_size, 1)) * (width_offset / (8 - width))
        transformations = {"crop_tops": tops, "crop_lefts": lefts}
        layer = layers.RandomCrop(height, width)
        with unittest.mock.patch.object(
            layer,
            "get_random_transformation_batch",
            return_value=transformations,
        ):
            inp = np.random.random((12, 5, 8, 3))
            actual_output = layer(inp)
            expected_output = inp[
                :,
                height_offset : (height_offset + height),
                width_offset : (width_offset + width),
                :,
            ]
            self.assertAllClose(expected_output, actual_output)

    def test_random_crop_full(self):
        np.random.seed(1337)
        height, width = 8, 16
        inp = np.random.random((12, 8, 16, 3))
        layer = layers.RandomCrop(height, width)
        actual_output = layer(inp)
        self.assertAllClose(inp, actual_output)

    def test_unbatched_image(self):
        np.random.seed(1337)
        inp = np.random.random((16, 16, 3))
        # manually compute transformations which shift 2 pixels
        mock_offset = tf.ones(shape=(1, 1), dtype=tf.float32) * 0.25
        layer = layers.RandomCrop(8, 8)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_offset,
        ):
            actual_output = layer(inp)
            self.assertAllClose(inp[2:10, 2:10, :], actual_output)

    def test_batched_input(self):
        np.random.seed(1337)
        inp = np.random.random((20, 16, 16, 3))
        # manually compute transformations which shift 2 pixels
        mock_offset = tf.ones(shape=(20, 1), dtype=tf.float32) * 2 / (16 - 8)
        layer = layers.RandomCrop(8, 8)
        with unittest.mock.patch.object(
            layer._random_generator,
            "random_uniform",
            return_value=mock_offset,
        ):
            actual_output = layer(inp)
            self.assertAllClose(inp[:, 2:10, 2:10, :], actual_output)

    def test_augment_bounding_boxes_crop(self):
        orig_height, orig_width = 512, 512
        height, width = 100, 200
        input_image = np.random.random((orig_height, orig_width, 3)).astype(
            np.float32
        )
        bboxes = {
            "boxes": tf.convert_to_tensor([[200, 200, 400, 400]]),
            "classes": tf.convert_to_tensor([1]),
        }
        input = {"images": input_image, "bounding_boxes": bboxes}
        # for top = 300 and left = 305
        height_offset = 300
        width_offset = 305
        tops = tf.ones((1, 1)) * (height_offset / (orig_height - height))
        lefts = tf.ones((1, 1)) * (width_offset / (orig_width - width))
        transformations = {"crop_tops": tops, "crop_lefts": lefts}
        layer = layers.RandomCrop(
            height=height, width=width, bounding_box_format="xyxy"
        )
        with unittest.mock.patch.object(
            layer,
            "get_random_transformation_batch",
            return_value=transformations,
        ):
            output = layer(input)
            expected_output = np.asarray(
                [[0.0, 0.0, 95.0, 100.0]],
            )
        self.assertAllClose(expected_output, output["bounding_boxes"]["boxes"])

    def test_augment_bounding_boxes_resize(self):
        input_image = np.random.random((256, 256, 3)).astype(np.float32)
        bboxes = {
            "boxes": tf.convert_to_tensor([[100, 100, 200, 200]]),
            "classes": tf.convert_to_tensor([1]),
        }
        input = {"images": input_image, "bounding_boxes": bboxes}
        layer = layers.RandomCrop(
            height=512, width=512, bounding_box_format="xyxy"
        )
        output = layer(input)
        expected_output = np.asarray(
            [[200.0, 200.0, 400.0, 400.0]],
        )
        self.assertAllClose(expected_output, output["bounding_boxes"]["boxes"])

    def test_random_crop_on_batched_ragged_images_and_bounding_boxes(self):
        images = tf.ragged.constant(
            [np.ones((8, 8, 3)), np.ones((4, 8, 3))], dtype="float32"
        )
        boxes = {
            "boxes": tf.ragged.stack(
                [
                    tf.ones((3, 4), dtype=tf.float32),
                    tf.ones((3, 4), dtype=tf.float32),
                ],
            ),
            "classes": tf.ragged.stack(
                [
                    tf.ones((3,), dtype=tf.float32),
                    tf.ones((3,), dtype=tf.float32),
                ],
            ),
        }
        inputs = {"images": images, "bounding_boxes": boxes}
        layer = layers.RandomCrop(height=2, width=2, bounding_box_format="xyxy")

        results = layer(inputs)

        self.assertTrue(isinstance(results["images"], tf.Tensor))
        self.assertTrue(
            isinstance(results["bounding_boxes"]["boxes"], tf.RaggedTensor)
        )
        self.assertTrue(
            isinstance(results["bounding_boxes"]["classes"], tf.RaggedTensor)
        )

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        layer = layers.RandomCrop(height=2, width=2)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (2, 2))
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
        layer = layers.RandomCrop(height=2, width=2)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (2, 2))
        self.assertAllInSet(result["segmentation_masks"], tf.range(0, 10))

    def test_ragged_input_with_graph_mode(self):
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
        layer = layers.RandomCrop(height=8, width=8)

        @tf.function
        def fn(inputs):
            outputs = layer(inputs)
            image_shape = outputs["images"].shape
            segmentation_mask_shape = outputs["segmentation_masks"].shape
            assert image_shape == (2, 8, 8, 3)
            assert segmentation_mask_shape == (2, 8, 8, 1)

        fn({"images": images, "segmentation_masks": segmentation_masks})
