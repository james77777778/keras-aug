import numpy as np
import tensorflow as tf
from keras_cv import bounding_box

from keras_aug import augmentations


class ResizeBySmallestSideTest(tf.test.TestCase):
    min_size = 224
    height, width = 224, 448
    regular_args = {
        "min_size": [224],
        "interpolation": "bilinear",
        "antialias": False,
        "bounding_box_format": "rel_xyxy",
    }

    def test_with_uint8(self):
        image_shape = (4, self.min_size, self.min_size, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = augmentations.ResizeBySmallestSide(**self.regular_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

    def test_config_with_custom_name(self):
        layer = augmentations.ResizeBySmallestSide(
            **self.regular_args, name="image_preproc"
        )

        config = layer.get_config()
        layer_reconstructed = augmentations.ResizeBySmallestSide.from_config(
            config
        )

        self.assertEqual(layer_reconstructed.name, layer.name)

    def test_config(self):
        layer = augmentations.ResizeBySmallestSide(**self.regular_args)

        config = layer.get_config()

        self.assertEqual(
            config["min_size"],
            self.regular_args["min_size"],
        )
        self.assertEqual(
            config["interpolation"], self.regular_args["interpolation"]
        )
        self.assertEqual(config["antialias"], self.regular_args["antialias"])
        self.assertEqual(
            config["bounding_box_format"],
            self.regular_args["bounding_box_format"],
        )

    def test_output_dtypes(self):
        inputs = tf.random.uniform(
            (self.height, self.width, 3), dtype=tf.float64
        )
        layer = augmentations.ResizeBySmallestSide(**self.regular_args)

        self.assertAllEqual(layer(inputs).dtype, "float32")

        layer = augmentations.ResizeBySmallestSide(
            **self.regular_args, dtype="uint8"
        )

        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_no_adjustment(self):
        image_shape = (4, self.min_size, self.min_size, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.ResizeBySmallestSide(**self.regular_args)
        output = layer(image)
        self.assertAllClose(image, output)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, self.min_size, self.min_size, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = augmentations.ResizeBySmallestSide(**self.regular_args)
        output = layer(image)
        self.assertAllClose(image, output)

    def test_resize_image(self):
        input_image_shape = (4, self.height, self.width, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = augmentations.ResizeBySmallestSide(
            **self.regular_args, seed=2023
        )

        output = layer(image)

        self.assertAllEqual(
            output.shape, (4, self.min_size, self.min_size * 2, 3)
        )

    def test_resize_smallest_image(self):
        input_image_shape = (123, 456, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = augmentations.ResizeBySmallestSide(
            **self.regular_args, seed=2023
        )

        output = layer(image)

        ratio = self.min_size / 123
        larger_side = round(456 * ratio)
        self.assertAllEqual(tf.reduce_max(output.shape[:2]), larger_side)
        self.assertAllEqual(tf.reduce_min(output.shape[:2]), self.min_size)

    def test_resize_smallest2_image(self):
        input_image_shape = (567, 456, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = augmentations.ResizeBySmallestSide(
            **self.regular_args, seed=2023
        )

        output = layer(image)

        ratio = self.min_size / 456
        larger_side = round(567 * ratio)
        self.assertAllEqual(tf.reduce_max(output.shape[:2]), larger_side)
        self.assertAllEqual(tf.reduce_min(output.shape[:2]), self.min_size)

    def test_grayscale(self):
        input_image_shape = (4, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)
        layer = augmentations.ResizeBySmallestSide(
            **self.regular_args, seed=2023
        )

        output = layer(image)

        self.assertAllEqual(
            output.shape, (4, self.min_size, self.min_size * 2, 1)
        )

    def test_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, 300, 300, 3)
        mask_shape = (1, 300, 300, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.regular_args.copy()
        args.update({"min_size": 150})
        layer = augmentations.ResizeBySmallestSide(**args, seed=2023)
        input_mask_resized = tf.image.resize(mask, (150, 150), "nearest")

        output = layer(inputs)

        self.assertAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.regular_args.copy()
        args.update({"min_size": 233})
        layer = augmentations.ResizeBySmallestSide(**args, seed=2023)

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
        args.update({"min_size": 10})
        layer = augmentations.ResizeBySmallestSide(**args, seed=2023)
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
        args.update({"min_size": 18})
        layer = augmentations.ResizeBySmallestSide(**args, seed=2023)
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
        args.update({"min_size": 18})
        layer = augmentations.ResizeBySmallestSide(**args, seed=2023)
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
