import tensorflow as tf
from absl.testing import parameterized
from keras_cv import bounding_box

from keras_aug import layers


class RandomZoomAndCropTest(tf.test.TestCase, parameterized.TestCase):
    batch_size = 4
    ori_height = 9
    ori_width = 8
    seed = 13
    height = 4
    width = 4

    def test_train_augments_image(self):
        # Checks if original and augmented images are different

        input_image_shape = (
            self.batch_size,
            self.ori_height,
            self.ori_width,
            3,
        )
        image = tf.random.uniform(shape=input_image_shape, seed=self.seed)

        layer = layers.RandomZoomAndCrop(
            height=self.height,
            width=self.width,
            scale_factor=(3 / 4, 4 / 3),
            seed=self.seed,
        )
        output = layer(image)

        input_image_resized = tf.image.resize(image, (self.height, self.width))

        self.assertNotAllClose(output, input_image_resized)

    def test_augment_bounding_box_single(self):
        image = tf.zeros([20, 20, 3])
        boxes = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }
        input = {"images": image, "bounding_boxes": boxes}

        layer = layers.RandomZoomAndCrop(
            height=self.height,
            width=self.width,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )
        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )
        expected_output = {
            "boxes": tf.convert_to_tensor([[0, 0, 1, 1]], dtype=tf.float32),
            "classes": tf.convert_to_tensor([0], dtype=tf.float32),
        }
        self.assertAllClose(
            expected_output["boxes"],
            output["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_augment_boxes_batched_input(self):
        image = tf.zeros([20, 20, 3])

        bounding_boxes = {
            "classes": tf.convert_to_tensor([[0, 0], [0, 0]]),
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ]
            ),
        }
        input = {"images": [image, image], "bounding_boxes": bounding_boxes}

        layer = layers.RandomZoomAndCrop(
            height=self.height,
            width=self.width,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )
        output = layer(input, training=True)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )
        expected_output = {
            "classes": tf.convert_to_tensor([[0, 0], [0, 0]], dtype=tf.float32),
            "boxes": tf.convert_to_tensor(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ],
                dtype=tf.float32,
            ),
        }
        self.assertAllClose(
            expected_output["boxes"],
            output["bounding_boxes"]["boxes"],
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
            "classes": tf.ragged.constant(
                [[0, 0], [0]],
                dtype=tf.float32,
            ),
        }
        input = {"images": image, "bounding_boxes": boxes}

        layer = layers.RandomZoomAndCrop(
            height=self.height,
            width=self.width,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )
        output = layer(input)
        # the result boxes will still have the entire image in them
        expected_output = {
            "boxes": tf.ragged.constant(
                [[[0, 0, 1, 1], [0, 0, 1, 1]], [[0, 0, 1, 1]]], dtype=tf.float32
            ),
            "classes": tf.ragged.constant(
                [[0.0, 0.0], [0.0]],
                dtype=tf.float32,
            ),
        }
        self.assertAllClose(
            expected_output["boxes"].to_tensor(),
            output["bounding_boxes"]["boxes"].to_tensor(),
        )
        self.assertAllClose(
            expected_output["classes"], output["bounding_boxes"]["classes"]
        )

    def test_dense_segmentation_masks(self):
        images = tf.random.uniform((2, 10, 10, 3))
        segmentation_masks = tf.random.uniform(
            (2, 10, 10, 1), minval=0, maxval=10, dtype=tf.int32
        )
        layer = layers.RandomZoomAndCrop(
            height=self.height,
            width=self.width,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(
            result["segmentation_masks"].shape[1:3], (self.height, self.width)
        )
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
        layer = layers.RandomZoomAndCrop(
            height=self.height,
            width=self.width,
            scale_factor=(3 / 4, 4 / 3),
            bounding_box_format="rel_xyxy",
            seed=self.seed,
        )

        result = layer(
            {"images": images, "segmentation_masks": segmentation_masks}
        )

        self.assertTrue(isinstance(result["segmentation_masks"], tf.Tensor))
        self.assertEqual(
            result["segmentation_masks"].shape[1:3], (self.height, self.width)
        )
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
        layer = layers.RandomZoomAndCrop(
            height=8, width=8, scale_factor=(3 / 4, 4 / 3)
        )

        @tf.function
        def fn(inputs):
            outputs = layer(inputs)
            image_shape = outputs["images"].shape
            segmentation_mask_shape = outputs["segmentation_masks"].shape
            assert image_shape == (2, 8, 8, 3)
            assert segmentation_mask_shape == (2, 8, 8, 1)

        fn({"images": images, "segmentation_masks": segmentation_masks})
