import numpy as np
import tensorflow as tf
from keras_cv import bounding_box

from keras_aug import layers


class CenterCropTest(tf.test.TestCase):
    height, width = 224, 224
    regular_args = {
        "height": 112,
        "width": 112,
        "bounding_box_format": "rel_xyxy",
    }
    no_aug_args = {
        "height": 224,
        "width": 224,
        "bounding_box_format": "rel_xyxy",
    }

    def test_no_adjustment(self):
        image_shape = (4, self.height, self.width, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = layers.CenterCrop(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, self.height, self.width, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = layers.CenterCrop(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = layers.CenterCrop(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_augments_image(self):
        input_image_shape = (4, self.height, self.width, 3)
        image = tf.random.uniform(shape=input_image_shape)
        layer = layers.CenterCrop(**self.regular_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertNotAllClose(output, input_image_resized)

    def test_grayscale(self):
        input_image_shape = (4, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape)

        layer = layers.CenterCrop(**self.regular_args, seed=2023)
        input_image_resized = tf.image.resize(image, [self.height, self.width])

        output = layer(image)

        self.assertAllEqual(
            output.shape,
            (4, self.regular_args["height"], self.regular_args["width"], 1),
        )
        self.assertNotAllClose(output, input_image_resized)

    def test_augment_sparse_segmentation_mask(self):
        num_classes = 8
        input_image_shape = (1, self.height, self.width, 3)
        mask_shape = (1, self.height, self.width, 1)
        image = tf.random.uniform(shape=input_image_shape, seed=2023)
        mask = np.random.randint(2, size=mask_shape) * (num_classes - 1)
        inputs = {"images": image, "segmentation_masks": mask}

        # Crop-only to exactly 1/2 of the size
        args = self.no_aug_args.copy()
        args.update({"height": 112, "width": 112})
        layer = layers.CenterCrop(**args)
        input_mask_resized = tf.image.crop_and_resize(
            mask, [[0, 0, 1, 1]], [0], (112, 112), "nearest"
        )

        output = layer(inputs)

        self.assertNotAllClose(output["segmentation_masks"], input_mask_resized)

        # Crop to an arbitrary size and make sure we don't do bad interpolation
        args = self.no_aug_args.copy()
        args.update({"height": 123, "width": 123})
        layer = layers.CenterCrop(**args)

        output = layer(inputs)

        self.assertAllInSet(output["segmentation_masks"], [0, num_classes - 1])

    def test_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]]),
            "classes": tf.convert_to_tensor([0]),
        }
        input = {"images": image, "bounding_boxes": boxes}
        args = self.no_aug_args.copy()
        args.update({"height": 10, "width": 10})
        layer = layers.CenterCrop(**args)
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
        args = self.no_aug_args.copy()
        args.update({"height": 18, "width": 18})
        layer = layers.CenterCrop(**args)
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
        args = self.no_aug_args.copy()
        args.update({"height": 18, "width": 18})
        layer = layers.CenterCrop(**args)
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

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        args = self.no_aug_args.copy()
        args.update({"height": 4, "width": 4})
        layer = layers.CenterCrop(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (4, 4))
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
        args.update({"height": 4, "width": 4})
        layer = layers.CenterCrop(**args)

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(result["segmentation_masks"].shape[1:3], (4, 4))
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
        layer = layers.CenterCrop(**args)

        @tf.function
        def fn(inputs):
            outputs = layer(inputs)
            image_shape = outputs["images"].shape
            segmentation_mask_shape = outputs["segmentation_masks"].shape
            assert image_shape == (2, 8, 8, 3)
            assert segmentation_mask_shape == (2, 8, 8, 1)

        fn({"images": images, "segmentation_masks": segmentation_masks})
