import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.center_crop import CenterCrop


class CenterCropTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_correctness(self):
        import torch
        import torchvision.transforms.v2.functional as TF

        # Test channels_last
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        layer = CenterCrop(16)
        y = layer(x)

        ref_y = TF.center_crop(torch.tensor(np.transpose(x, [0, 3, 1, 2])), 16)
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        layer = CenterCrop(16)
        y = layer(x)

        ref_y = TF.center_crop(torch.tensor(x), 16)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

    def test_correctness_uint8(self):
        import torch
        import torchvision.transforms.v2.functional as TF

        # Test channels_last
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("uint8")
        layer = CenterCrop(16)
        y = layer(x)

        ref_y = TF.center_crop(torch.tensor(np.transpose(x, [0, 3, 1, 2])), 16)
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 255, (2, 3, 32, 32)).astype("uint8")
        layer = CenterCrop(16)
        y = layer(x)

        ref_y = TF.center_crop(torch.tensor(x), 16)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)

    @parameterized.named_parameters(
        named_product(mode=["constant", "reflect", "symmetric"])
    )
    def test_mode(self, mode):
        if backend.backend() == "torch" and mode == "symmetric":
            self.skipTest("TODO: Need to investigate")
        np.random.seed(42)
        x = np.random.uniform(0.5, 1, (1, 32, 32, 3)).astype("float32")
        layer = CenterCrop(48, padding_mode=mode)
        y = layer(x)

        pad_width = [[0, 0], [8, 8], [8, 8], [0, 0]]
        ref_y = np.pad(x, pad_width, mode=mode)
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = CenterCrop(16)(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = CenterCrop(16)(x)
        self.assertEqual(y.shape, (None, 3, 16, 16))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = CenterCrop(16)(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

    def test_model(self):
        layer = CenterCrop(16)
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 16, 16, 3))

    def test_config(self):
        x = np.random.uniform(0, 255, (2, 32, 32, 3)).astype("float32")
        layer = CenterCrop(16)
        y = layer(x)

        layer = CenterCrop.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = CenterCrop(16)
        x = np.random.uniform(size=(3, 32, 32, 3)).astype("float32") * 255
        ds = tf.data.Dataset.from_tensor_slices(x).batch(3).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (3, 16, 16, 3))

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
        input = {"images": images, "bounding_boxes": boxes}
        layer = CenterCrop((18, 18), bounding_box_format="rel_xyxy")

        output = layer(input)
        self.assertAllClose(output["bounding_boxes"]["boxes"], boxes["boxes"])
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
        input = {"images": images, "bounding_boxes": boxes}
        layer = CenterCrop((18, 18), bounding_box_format="xyxy")

        output = layer(input)
        expected_boxes = {
            "boxes": np.array(
                [
                    [[2, 3, 16, 17], [9, 11, 15, 18]],
                    [[0, 0, 0, 0], [14, 11, 16, 18]],
                ],
                "float32",
            ),
            "classes": np.array([[0, 1], [-1, 3]], "float32"),
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
        masks = np.random.randint(2, size=masks_shape) * (num_classes - 1)
        inputs = {"images": images, "segmentation_masks": masks}

        # Crop to exactly 1/2 of the size
        ref_masks = masks[:, 8:-8, 8:-8, :]
        layer = CenterCrop(16)
        output = layer(inputs)
        self.assertAllClose(output["segmentation_masks"], ref_masks)
