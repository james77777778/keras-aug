import numpy as np
import tensorflow as tf
from keras_cv import bounding_box

from keras_aug import augmentations


class RandomRotationTest(tf.test.TestCase):
    regular_args = {
        "rotation_factor": 10 / 360,
        "translation_height_factor": 0.1,
        "translation_width_factor": 0.1,
        "zoom_height_factor": 0.1,
        "zoom_width_factor": 0.1,
        "shear_height_factor": 0.1,
        "shear_width_factor": 0.1,
        "fill_mode": "constant",
        "fill_value": 0,
        "interpolation": "bilinear",
        "bounding_box_format": "xyxy",
    }
    no_aug_args = {
        "rotation_factor": 0.0,
        "translation_height_factor": 0.0,
        "translation_width_factor": 0.0,
        "zoom_height_factor": 0.0,
        "zoom_width_factor": 0.0,
        "shear_height_factor": 0.0,
        "shear_width_factor": 0.0,
        "fill_mode": "constant",
        "fill_value": 0,
        "interpolation": "bilinear",
        "bounding_box_format": "xyxy",
    }

    def test_preserves_output_shape(self):
        input_images = np.random.random((2, 8, 8, 3)).astype(np.float32)
        expected_output = input_images
        layer = augmentations.RandomAffine(**self.regular_args)

        actual_output = layer(input_images)

        self.assertEqual(expected_output.shape, actual_output.shape)

    def test_with_uint8(self):
        image_shape = (4, 8, 8, 3)
        image = tf.cast(
            tf.random.uniform(shape=image_shape) * 255.0, dtype=tf.uint8
        )

        layer = augmentations.RandomAffine(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = augmentations.RandomAffine(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_independence_on_batched_images(self):
        image = tf.random.uniform((100, 100, 3))
        batched_images = tf.stack((image, image), axis=0)
        layer = augmentations.RandomAffine(**self.regular_args, seed=2023)

        results = layer(batched_images)

        self.assertNotAllClose(results[0], results[1])

    def test_config_with_custom_name(self):
        layer = augmentations.RandomAffine(
            **self.regular_args, name="image_preproc"
        )

        config = layer.get_config()
        layer_reconstructed = augmentations.RandomAffine.from_config(config)

        self.assertEqual(layer_reconstructed.name, layer.name)

    def test_config(self):
        layer = augmentations.RandomAffine(**self.regular_args)

        config = layer.get_config()

        self.assertEqual(
            config["translation_height_factor"],
            self.regular_args["translation_height_factor"],
        )
        self.assertEqual(
            config["translation_width_factor"],
            self.regular_args["translation_width_factor"],
        )
        self.assertEqual(
            config["zoom_height_factor"],
            self.regular_args["zoom_height_factor"],
        )
        self.assertEqual(
            config["zoom_width_factor"], self.regular_args["zoom_width_factor"]
        )
        self.assertEqual(
            config["shear_height_factor"],
            self.regular_args["shear_height_factor"],
        )
        self.assertEqual(
            config["shear_width_factor"],
            self.regular_args["shear_width_factor"],
        )
        self.assertEqual(config["fill_mode"], self.regular_args["fill_mode"])
        self.assertEqual(config["fill_value"], self.regular_args["fill_value"])
        self.assertEqual(
            config["interpolation"], self.regular_args["interpolation"]
        )
        self.assertEqual(
            config["bounding_box_format"],
            self.regular_args["bounding_box_format"],
        )

    def test_output_dtypes(self):
        inputs = np.array([[[1], [2]], [[3], [4]]], dtype="float64")
        layer = augmentations.RandomAffine(**self.regular_args)

        self.assertAllEqual(layer(inputs).dtype, "float32")

        layer = augmentations.RandomAffine(**self.regular_args, dtype="uint8")

        self.assertAllEqual(layer(inputs).dtype, "uint8")

    def test_no_adjustment(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.RandomAffine(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, atol=1e-5, rtol=1e-5)

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0

        layer = augmentations.RandomAffine(**self.no_aug_args)
        output = layer(image)
        self.assertAllClose(image, output, rtol=1e-5, atol=1e-5)

        layer = augmentations.RandomAffine(**self.regular_args)
        output = layer(image)
        self.assertNotAllClose(image, output)

    def test_rotation_unbatched_image(self):
        input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(np.float32)
        # 180 rotation
        args = self.no_aug_args.copy()
        args.update({"rotation_factor": (0.5, 0.5)})
        layer = augmentations.RandomAffine(**args)
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

    def test_rotation_bounding_boxes(self):
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
        args.update({"rotation_factor": (0.5, 0.5)})
        layer = augmentations.RandomAffine(**args)

        output = layer(input)
        output["bounding_boxes"] = bounding_box.to_dense(
            output["bounding_boxes"]
        )

        self.assertAllClose(expected_bounding_boxes, output["bounding_boxes"])

    def test_rotation_ragged_bounding_boxes(self):
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
        args.update({"rotation_factor": (0.5, 0.5)})
        layer = augmentations.RandomAffine(**args)

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

    def test_rotation_sparse_segmentation_mask(self):
        num_classes = 8

        input_images = np.random.random((2, 20, 20, 3)).astype(np.float32)
        # Masks are all 0s or 8s, to verify that when we rotate we don't do bad
        # mask interpolation to either a 0 or a 7
        masks = np.random.randint(2, size=(2, 20, 20, 1)) * (num_classes - 1)
        inputs = {"images": input_images, "segmentation_masks": masks}

        # 90 rotation
        args = self.no_aug_args.copy()
        args.update({"rotation_factor": (0.25, 0.25)})
        layer = augmentations.RandomAffine(**args)
        outputs = layer(inputs)
        expected_masks = np.rot90(masks, axes=(1, 2))
        self.assertAllClose(expected_masks, outputs["segmentation_masks"])

        # 45-degree rotation. Only verifies that no interpolation takes place.
        # 90 rotation
        args = self.no_aug_args.copy()
        args.update({"rotation_factor": (0.125, 0.125)})
        outputs = layer(inputs)
        self.assertAllInSet(outputs["segmentation_masks"], [0, 7])

    def test_translation_up_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(
                dtype
            )
            # Shifting by -.2 * 5 = 1 pixel.
            args = self.no_aug_args.copy()
            args.update({"translation_height_factor": (-0.2, -0.2)})
            layer = augmentations.RandomAffine(**args)
            expected_output = (
                np.asarray(
                    [
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                        [20, 21, 22, 23, 24],
                        [0, 0, 0, 0, 0],
                    ]
                )
                .astype(dtype)
                .reshape((1, 5, 5, 1))
            )

            output_image = layer(input_image)

            self.assertAllEqual(expected_output, output_image)

    def test_translation_down_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(
                dtype
            )
            # Shifting by -.2 * 5 = 1 pixel.
            args = self.no_aug_args.copy()
            args.update({"translation_height_factor": (0.2, 0.2)})
            layer = augmentations.RandomAffine(**args)
            expected_output = (
                np.asarray(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 1, 2, 3, 4],
                        [5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14],
                        [15, 16, 17, 18, 19],
                    ]
                )
                .astype(dtype)
                .reshape((1, 5, 5, 1))
            )

            output_image = layer(input_image)

            self.assertAllEqual(expected_output, output_image)

    def test_translation_left_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(
                dtype
            )
            # Shifting by -.2 * 5 = 1 pixel.
            args = self.no_aug_args.copy()
            args.update({"translation_width_factor": (-0.2, -0.2)})
            layer = augmentations.RandomAffine(**args)
            expected_output = (
                np.asarray(
                    [
                        [1, 2, 3, 4, 0],
                        [6, 7, 8, 9, 0],
                        [11, 12, 13, 14, 0],
                        [16, 17, 18, 19, 0],
                        [21, 22, 23, 24, 0],
                    ]
                )
                .astype(dtype)
                .reshape((1, 5, 5, 1))
            )

            output_image = layer(input_image)

            self.assertAllEqual(expected_output, output_image)

    def test_translation_right_constant(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (1, 5, 5, 1)).astype(
                dtype
            )
            # Shifting by -.2 * 5 = 1 pixel.
            args = self.no_aug_args.copy()
            args.update({"translation_width_factor": (0.2, 0.2)})
            layer = augmentations.RandomAffine(**args)
            expected_output = (
                np.asarray(
                    [
                        [0, 0, 1, 2, 3],
                        [0, 5, 6, 7, 8],
                        [0, 10, 11, 12, 13],
                        [0, 15, 16, 17, 18],
                        [0, 20, 21, 22, 23],
                    ]
                )
                .astype(dtype)
                .reshape((1, 5, 5, 1))
            )

            output_image = layer(input_image)

            self.assertAllEqual(expected_output, output_image)

    def test_zoom_in_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(dtype)
            args = self.no_aug_args.copy()
            args.update(
                {
                    "zoom_height_factor": (-0.5, -0.5),
                    "zoom_width_factor": (-0.5, -0.5),
                    "interpolation": "nearest",
                }
            )
            layer = augmentations.RandomAffine(**args)
            expected_output = (
                np.asarray(
                    [
                        [6, 7, 7, 8, 8],
                        [11, 12, 12, 13, 13],
                        [11, 12, 12, 13, 13],
                        [16, 17, 17, 18, 18],
                        [16, 17, 17, 18, 18],
                    ]
                )
                .astype(dtype)
                .reshape((1, 5, 5, 1))
            )

            output_image = layer(np.expand_dims(input_image, axis=0))

            self.assertAllEqual(expected_output, output_image)

    def test_zoom_out_numeric(self):
        for dtype in (np.int64, np.float32):
            input_image = np.reshape(np.arange(0, 25), (5, 5, 1)).astype(dtype)
            args = self.no_aug_args.copy()
            args.update(
                {
                    "zoom_height_factor": (0.5, 0.5),
                    "zoom_width_factor": (0.8, 0.8),
                    "interpolation": "nearest",
                }
            )
            layer = augmentations.RandomAffine(**args)
            expected_output = (
                np.asarray(
                    [
                        [0, 0, 0, 0, 0],
                        [0, 5, 7, 9, 0],
                        [0, 10, 12, 14, 0],
                        [0, 20, 22, 24, 0],
                        [0, 0, 0, 0, 0],
                    ]
                )
                .astype(dtype)
                .reshape((1, 5, 5, 1))
            )

            output_image = layer(np.expand_dims(input_image, axis=0))

            self.assertAllEqual(expected_output, output_image)

    def test_shear_area(self):
        xs = tf.ones((1, 512, 512, 3))
        ys = {
            "boxes": tf.constant(
                [[[0.3, 0.4, 0.5, 0.6], [0.9, 0.8, 1.0, 1.0]]]
            ),
            "classes": tf.constant([[2, 3]]),
        }

        inputs = {"images": xs, "bounding_boxes": ys}
        args = self.no_aug_args.copy()
        args.update(
            {
                "shear_height_factor": (0.3, 0.7),
                "shear_width_factor": (0.4, 0.7),
                "interpolation": "nearest",
                "seed": 0,
                "bounding_box_format": "rel_xyxy",
            }
        )
        layer = augmentations.RandomAffine(**args)

        outputs = layer(inputs)
        xs, ys_bounding_boxes = (
            outputs["images"],
            outputs["bounding_boxes"]["boxes"],
        )
        new_area = tf.math.multiply(
            tf.abs(
                tf.subtract(
                    ys_bounding_boxes[..., 2], ys_bounding_boxes[..., 0]
                )
            ),
            tf.abs(
                tf.subtract(
                    ys_bounding_boxes[..., 3], ys_bounding_boxes[..., 1]
                )
            ),
        )
        old_area = tf.math.multiply(
            tf.abs(tf.subtract(ys["boxes"][..., 2], ys["boxes"][..., 0])),
            tf.abs(tf.subtract(ys["boxes"][..., 3], ys["boxes"][..., 1])),
        )

        self.assertTrue(tf.math.reduce_all(new_area > old_area))
