import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras_cv import bounding_box

from keras_aug import layers


class ResizeTest(tf.test.TestCase, parameterized.TestCase):
    height, width = 224, 224
    regular_args = {
        "height": 224,
        "width": 224,
        "interpolation": "bilinear",
        "antialias": False,
        "bounding_box_format": "rel_xyxy",
    }
    resize_with_crop_args = {
        "height": 224,
        "width": 224,
        "interpolation": "bilinear",
        "antialias": False,
        "crop_to_aspect_ratio": True,
        "position": "center",
        "padding_value": 0,
        "bounding_box_format": "rel_xyxy",
    }
    resize_with_pad_args = {
        "height": 224,
        "width": 224,
        "interpolation": "bilinear",
        "antialias": False,
        "pad_to_aspect_ratio": True,
        "position": "center",
        "padding_value": 0,
        "bounding_box_format": "rel_xyxy",
    }

    """Resize"""

    def test_augments_image(self):
        input_image_shape = (4, 200, 300, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.Resize(**self.regular_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertAllClose(output, input_image_resized)

    def test_grayscale(self):
        input_image_shape = (4, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.Resize(**self.regular_args, seed=2023)

        output = layer(image)

        self.assertAllEqual(output.shape, (4, self.height, self.width, 1))

    def test_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, 300, 300, 3)
        mask_shape = (1, 300, 300, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.regular_args.copy()
        args.update({"height": 150, "width": 150})
        layer = layers.Resize(**args, seed=2023)
        input_mask_resized = tf.image.resize(mask, (150, 150), "nearest")

        output = layer(inputs)

        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.regular_args.copy()
        args.update({"height": 233, "width": 233})
        layer = layers.Resize(**args, seed=2023)

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
        args.update({"height": 10, "width": 10})
        layer = layers.Resize(**args, seed=2023)
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
        args.update({"height": 18, "width": 18})
        layer = layers.Resize(**args, seed=2023)
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
        args.update({"height": 18, "width": 18})
        layer = layers.Resize(**args, seed=2023)
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

    def _run_output_shape_test(self, kwargs, height, width):
        kwargs.update({"height": height, "width": width})
        layer = layers.Resize(**kwargs)
        inputs = tf.random.uniform((2, 5, 8, 3))

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (2, height, width, 3))

    @parameterized.named_parameters(
        ("down_sample_bilinear_2_by_2", {"interpolation": "bilinear"}, 2, 2),
        ("down_sample_bilinear_3_by_2", {"interpolation": "bilinear"}, 3, 2),
        ("down_sample_nearest_2_by_2", {"interpolation": "nearest"}, 2, 2),
        ("down_sample_nearest_3_by_2", {"interpolation": "nearest"}, 3, 2),
        ("down_sample_area_2_by_2", {"interpolation": "area"}, 2, 2),
        ("down_sample_area_3_by_2", {"interpolation": "area"}, 3, 2),
    )
    def test_down_sampling(self, kwargs, height, width):
        self._run_output_shape_test(kwargs, height, width)

    @parameterized.named_parameters(
        ("up_sample_bilinear_10_by_12", {"interpolation": "bilinear"}, 10, 12),
        ("up_sample_bilinear_12_by_12", {"interpolation": "bilinear"}, 12, 12),
        ("up_sample_nearest_10_by_12", {"interpolation": "nearest"}, 10, 12),
        ("up_sample_nearest_12_by_12", {"interpolation": "nearest"}, 12, 12),
        ("up_sample_area_10_by_12", {"interpolation": "area"}, 10, 12),
        ("up_sample_area_12_by_12", {"interpolation": "area"}, 12, 12),
    )
    def test_up_sampling(self, kwargs, expected_height, expected_width):
        self._run_output_shape_test(kwargs, expected_height, expected_width)

    """Resize with crop_to_aspect_ratio"""

    def test_crop_augments_image(self):
        input_image_shape = (4, 200, 300, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.Resize(**self.resize_with_crop_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertNotAllClose(output, input_image_resized)

    def test_crop_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, 300, 300, 3)
        mask_shape = (1, 300, 300, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.resize_with_crop_args.copy()
        args.update({"height": 150, "width": 150})
        layer = layers.Resize(**args, seed=2023)
        input_mask_resized = tf.image.resize(mask, (150, 150), "nearest")

        output = layer(inputs)

        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.resize_with_crop_args.copy()
        args.update({"height": 233, "width": 233})
        layer = layers.Resize(**args, seed=2023)

        output = layer(inputs)

        self.assertAllInSet(output["segmentation_masks"], [0, 7])

    def test_crop_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]]),
            "classes": tf.convert_to_tensor([0]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.resize_with_crop_args.copy()
        args.update({"height": 10, "width": 10})
        layer = layers.Resize(**args, seed=2023)
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

    def test_crop_augment_boxes_batched_input(self):
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
        args = self.resize_with_crop_args.copy()
        args.update({"height": 18, "width": 18})
        layer = layers.Resize(**args, seed=2023)
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

    def test_crop_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant([[0, 0], [0]]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.resize_with_crop_args.copy()
        args.update({"height": 18, "width": 18})
        layer = layers.Resize(**args, seed=2023)
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

    def test_cropping_center(self):
        inputs = tf.random.uniform((1, 4, 8, 3))
        args = self.resize_with_crop_args.copy()
        args.update({"height": 4, "width": 4, "position": "center"})
        layer = layers.Resize(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 4, 4, 3))
        self.assertAllEqual(outputs, inputs[:, :, 2:-2, :])

    """Resize with pad_to_aspect_ratio"""

    def test_pad_augments_image(self):
        input_image_shape = (4, 200, 300, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.Resize(**self.resize_with_pad_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertNotAllClose(output, input_image_resized)

    def test_pad_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, 300, 300, 3)
        mask_shape = (1, 300, 300, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.resize_with_pad_args.copy()
        args.update({"height": 150, "width": 150})
        layer = layers.Resize(**args, seed=2023)
        input_mask_resized = tf.image.resize(mask, (150, 150), "nearest")

        output = layer(inputs)

        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.regular_args.copy()
        args.update({"height": 233, "width": 233})
        layer = layers.Resize(**args, seed=2023)

        output = layer(inputs)

        self.assertAllInSet(output["segmentation_masks"], [0, 7])

    def test_pad_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]]),
            "classes": tf.convert_to_tensor([0]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.resize_with_pad_args.copy()
        args.update({"height": 10, "width": 10})
        layer = layers.Resize(**args, seed=2023)
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

    def test_pad_augment_boxes_batched_input(self):
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
        args = self.resize_with_pad_args.copy()
        args.update({"height": 18, "width": 18})
        layer = layers.Resize(**args, seed=2023)
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

    def test_pad_augment_boxes_ragged(self):
        image = tf.zeros([2, 20, 20, 3])
        boxes = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant([[0, 0], [0]]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.resize_with_pad_args.copy()
        args.update({"height": 18, "width": 18})
        layer = layers.Resize(**args, seed=2023)
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

    def test_padding_center(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "center"})
        layer = layers.Resize(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, 0:2, :, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, -2:, :, :]), 0.0)

    def test_padding_top(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "top_left"})
        layer = layers.Resize(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertNotEqual(tf.reduce_mean(outputs[:, 0:2, :, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, -4:, :, :]), 0.0)

    def test_padding_bottom(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "bottom_left"})
        layer = layers.Resize(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, 0:4, :, :]), 0.0)
        self.assertNotEqual(tf.reduce_mean(outputs[:, -2:, :, :]), 0.0)

    def test_padding_left(self):
        inputs = tf.ones((1, 8, 4, 3))
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "top_left"})
        layer = layers.Resize(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertNotEqual(tf.reduce_mean(outputs[:, :, 0:2, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, :, -2:, :]), 0.0)

    def test_padding_right(self):
        inputs = tf.ones((1, 8, 4, 3))
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "top_right"})
        layer = layers.Resize(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, :, 0:4, :]), 0.0)
        self.assertNotEqual(tf.reduce_mean(outputs[:, :, -2:, :]), 0.0)

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "top_right"})
        layer = layers.Resize(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (8, 8))
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
        args = self.resize_with_pad_args.copy()
        args.update({"height": 8, "width": 8, "position": "top_right"})
        layer = layers.Resize(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (8, 8))
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
        args = self.regular_args.copy()
        args.update({"height": 8, "width": 8})
        layer = layers.Resize(**args)

        @tf.function
        def fn(inputs):
            outputs = layer(inputs)
            image_shape = outputs["images"].shape
            segmentation_mask_shape = outputs["segmentation_masks"].shape
            assert image_shape == (2, 8, 8, 3)
            assert segmentation_mask_shape == (2, 8, 8, 1)

        fn({"images": images, "segmentation_masks": segmentation_masks})
