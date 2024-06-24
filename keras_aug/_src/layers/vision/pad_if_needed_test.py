import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.pad_if_needed import PadIfNeeded
from keras_aug._src.utils.test_utils import get_images


class PadIfNeededTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    @parameterized.named_parameters(
        named_product(
            size=[(36, 36), (40, 50)],
            padding_position=[
                "border",
                "top_left",
                "top_right",
                "bottom_left",
                "bottom_right",
            ],
            padding_value=[0, 10],
            dtype=["float32", "uint8"],
        )
    )
    def test_correctness(self, size, padding_position, padding_value, dtype):
        np.random.seed(42)
        x = get_images(dtype, "channels_last")
        if dtype == "float32":
            x = np.clip(x, 0.5, 1.0)
        elif dtype == "uint8":
            x = np.clip(x, 127, 255)
        layer = PadIfNeeded(
            size, "constant", padding_position, padding_value, dtype=dtype
        )
        y = layer(x)

        self.assertEqual(tuple(y.shape), (2, *size, 3))
        if padding_position == "border":
            self.assertAllClose(y[0, 0, 0, 0], padding_value)
            self.assertAllClose(y[0, 0, -1, 0], padding_value)
            self.assertAllClose(y[0, -1, 0, 0], padding_value)
            self.assertAllClose(y[0, -1, -1, 0], padding_value)
        elif padding_position == "top_left":
            self.assertAllClose(y[0, 0, 0, 0], padding_value)
            self.assertAllClose(y[0, 0, -1, 0], padding_value)
            self.assertAllClose(y[0, -1, 0, 0], padding_value)
            self.assertNotAllClose(y[0, -1, -1, 0], padding_value)
        elif padding_position == "top_right":
            self.assertAllClose(y[0, 0, 0, 0], padding_value)
            self.assertAllClose(y[0, 0, -1, 0], padding_value)
            self.assertNotAllClose(y[0, -1, 0, 0], padding_value)
            self.assertAllClose(y[0, -1, -1, 0], padding_value)
        elif padding_position == "bottom_left":
            self.assertAllClose(y[0, 0, 0, 0], padding_value)
            self.assertNotAllClose(y[0, 0, -1, 0], padding_value)
            self.assertAllClose(y[0, -1, 0, 0], padding_value)
            self.assertAllClose(y[0, -1, -1, 0], padding_value)
        elif padding_position == "bottom_right":
            self.assertNotAllClose(y[0, 0, 0, 0], padding_value)
            self.assertAllClose(y[0, 0, -1, 0], padding_value)
            self.assertAllClose(y[0, -1, 0, 0], padding_value)
            self.assertAllClose(y[0, -1, -1, 0], padding_value)

    @parameterized.named_parameters(
        named_product(mode=["constant", "reflect", "symmetric"])
    )
    def test_mode(self, mode):
        if backend.backend() == "torch" and mode == "symmetric":
            self.skipTest("TODO: Need to investigate")
        np.random.seed(42)
        x = get_images("float32", "channels_last")
        x = np.clip(x, 0.5, 1.0)
        layer = PadIfNeeded(48, padding_mode=mode)
        y = layer(x)

        pad_width = [[0, 0], [8, 8], [8, 8], [0, 0]]
        ref_y = np.pad(x, pad_width, mode=mode)
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = PadIfNeeded(16)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = PadIfNeeded(48)(x)
        self.assertEqual(y.shape, (None, 48, 48, 3))

    def test_model(self):
        # Test dynamic shape
        layer = PadIfNeeded(16)
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

        # Test static shape
        layer = PadIfNeeded((32, 48))
        inputs = keras.layers.Input(shape=[32, 32, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 32, 48, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = PadIfNeeded((32, 48))
        y = layer(x)

        layer = PadIfNeeded.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = PadIfNeeded(48)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 48, 48, 3))

    def test_augment_bounding_box(self):
        # Test full bounding boxes
        images = np.zeros([1, 20, 20, 3]).astype("float32")
        boxes = {
            "boxes": np.array(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                    [[0, 0, 1, 1], [0, 0, 1, 1]],
                ],
                "float32",
            ),
            "classes": np.array([[0, 0], [0, 0]], "float32"),
        }
        input = {"images": images, "bounding_boxes": boxes.copy()}
        layer = PadIfNeeded(30, bounding_box_format="rel_xyxy")

        output = layer(input)
        self.assertNotAllClose(
            output["bounding_boxes"]["boxes"], boxes["boxes"]
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"], boxes["classes"]
        )

        # Test arbitrary bounding boxes
        images = np.zeros([1, 20, 20, 3]).astype("float32")
        boxes = {
            "boxes": np.array(
                [
                    [[3, 4, 17, 18], [10, 12, 16, 19]],
                    [[0, 0, 1, 1], [15, 12, 17, 20]],
                ],
                "float32",
            ),
            "classes": np.array([[0, 1], [2, 3]], "float32"),
        }
        input = {"images": images, "bounding_boxes": boxes.copy()}
        layer = PadIfNeeded(30, bounding_box_format="xyxy")

        output = layer(input)
        expected_boxes = {
            "boxes": np.array(
                [
                    [[8, 9, 22, 23], [15, 17, 21, 24]],
                    [[5, 5, 6, 6], [20, 17, 22, 25]],
                ],
                "float32",
            ),
            "classes": np.array([[0, 1], [2, 3]], "float32"),
        }
        self.assertAllClose(
            output["bounding_boxes"]["boxes"], expected_boxes["boxes"]
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"], expected_boxes["classes"]
        )

    def test_augment_segmentation_mask(self):
        num_classes = 8
        images_shape = (1, 32, 32, 3)
        masks_shape = (1, 32, 32, 1)
        images = np.random.uniform(size=images_shape).astype("float32")
        masks = np.random.randint(2, size=masks_shape) * (num_classes - 1) + 1
        inputs = {"images": images, "segmentation_masks": masks}

        layer = PadIfNeeded(48)
        output = layer(inputs)
        output_masks = output["segmentation_masks"][:, 8:-8, 8:-8, :]
        output_masks_border = output["segmentation_masks"][:, :8, :8, :]
        self.assertAllClose(output_masks, masks)
        self.assertAllClose(
            output_masks_border, np.ones_like(output_masks_border) * -1
        )
