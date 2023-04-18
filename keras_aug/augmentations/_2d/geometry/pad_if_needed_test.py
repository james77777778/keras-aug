import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras_cv import bounding_box

from keras_aug import augmentations


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

    def test_with_uint8(self):
        image_shape = (4, self.height, self.width, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = augmentations.PadIfNeeded(**self.regular_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

    def test_same_result_on_batched_images(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = augmentations.PadIfNeeded(**self.regular_args, seed=2023)

        results = layer(batched_images)

        self.assertAllClose(results[0], results[1])

    def test_config_with_custom_name(self):
        layer = augmentations.PadIfNeeded(
            **self.regular_args, name="image_preproc"
        )

        config = layer.get_config()
        layer_reconstructed = augmentations.PadIfNeeded.from_config(config)

        self.assertEqual(layer_reconstructed.name, layer.name)

    def test_config(self):
        layer = augmentations.PadIfNeeded(**self.regular_args)

        config = layer.get_config()

        self.assertEqual(
            config["min_height"],
            self.regular_args["min_height"],
        )
        self.assertEqual(
            config["min_width"],
            self.regular_args["min_width"],
        )
        self.assertEqual(
            config["pad_height_divisor"],
            self.regular_args["pad_height_divisor"],
        )
        self.assertEqual(
            config["pad_width_divisor"], self.regular_args["pad_width_divisor"]
        )
        # convert from PaddingPosition to str
        self.assertEqual(
            config["position"].value, self.regular_args["position"]
        )
        self.assertEqual(
            config["padding_value"], self.regular_args["padding_value"]
        )
        self.assertEqual(
            config["bounding_box_format"],
            self.regular_args["bounding_box_format"],
        )

    def test_output_dtypes(self):
        inputs = tf.random.uniform(
            (self.height, self.width, 3), dtype=tf.float64
        )
        layer = augmentations.PadIfNeeded(**self.regular_args)

        self.assertAllEqual(layer(inputs).dtype, "float32")

        layer = augmentations.PadIfNeeded(**self.regular_args, dtype="uint8")

        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_augments_image(self):
        input_image_shape = (4, 50, 60, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = augmentations.PadIfNeeded(**self.regular_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertNotAllClose(output, input_image_resized)

    def test_grayscale(self):
        input_image_shape = (4, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)
        layer = augmentations.PadIfNeeded(**self.regular_args, seed=2023)

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
        layer = augmentations.PadIfNeeded(**args, seed=2023)

        output = layer(inputs)

        self.assertAllClose(
            output["segmentation_masks"][:, 25:-25, 25:-25, :], mask
        )

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.regular_args.copy()
        args.update({"min_height": 123, "min_width": 123})
        layer = augmentations.PadIfNeeded(**args, seed=2023)

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
        layer = augmentations.PadIfNeeded(**args, seed=2023)
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
        layer = augmentations.PadIfNeeded(**args, seed=2023)
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
        layer = augmentations.PadIfNeeded(**args, seed=2023)
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
        layer = augmentations.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, 0:2, :, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, -2:, :, :]), 0.0)

    def test_padding_top(self):
        inputs = tf.ones((1, 4, 8, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "top_left"})
        layer = augmentations.PadIfNeeded(**args)

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
        layer = augmentations.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, 0:4, :, :]), 0.0)
        self.assertNotEqual(tf.reduce_mean(outputs[:, -2:, :, :]), 0.0)

    def test_padding_left(self):
        inputs = tf.ones((1, 8, 4, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "top_left"})
        layer = augmentations.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertNotEqual(tf.reduce_mean(outputs[:, :, 0:2, :]), 0.0)
        self.assertEqual(tf.reduce_mean(outputs[:, :, -2:, :]), 0.0)

    def test_padding_right(self):
        inputs = tf.ones((1, 8, 4, 3))
        args = self.regular_args.copy()
        args.update({"min_height": 8, "min_width": 8, "position": "top_right"})
        layer = augmentations.PadIfNeeded(**args)

        outputs = layer(inputs)

        self.assertEqual(outputs.shape, (1, 8, 8, 3))
        self.assertEqual(tf.reduce_mean(outputs[:, :, 0:4, :]), 0.0)
        self.assertNotEqual(tf.reduce_mean(outputs[:, :, -2:, :]), 0.0)
