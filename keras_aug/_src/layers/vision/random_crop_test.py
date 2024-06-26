import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_crop import RandomCrop
from keras_aug._src.utils.test_utils import get_images


class FixedRandomCrop(RandomCrop):
    def get_params(self, batch_size, images=None, **kwargs):
        return dict(
            pad_top=0,
            pad_bottom=0,
            pad_left=0,
            pad_right=0,
            crop_top=8,
            crop_left=10,
        )


class RandomCropTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_correctness(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        # Test channels_last
        x = get_images(dtype, "channels_last")
        layer = FixedRandomCrop(16, dtype=dtype)
        y = layer(x)

        ref_y = TF.crop(
            torch.tensor(np.transpose(x, [0, 3, 1, 2])), 8, 10, 16, 16
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images(dtype, "channels_first")
        layer = FixedRandomCrop(16, dtype=dtype)
        y = layer(x)

        ref_y = TF.crop(torch.tensor(x), 8, 10, 16, 16)
        ref_y = ref_y.cpu().numpy()
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y)

    @parameterized.named_parameters(
        named_product(mode=["constant", "reflect", "symmetric"])
    )
    @pytest.mark.skip("TODO: Investigate `mode`")
    def test_mode(self, mode):
        if backend.backend() == "torch" and mode == "symmetric":
            self.skipTest("TODO: Need to investigate")
        np.random.seed(42)
        x = get_images("float32", "channels_last")
        x = np.clip(x, 0.5, 1.0)
        layer = RandomCrop(48, padding_mode=mode)
        y = layer(x)

        pad_width = [[0, 0], [8, 8], [8, 8], [0, 0]]
        ref_y = np.pad(x, pad_width, mode=mode)
        self.assertAllClose(y, ref_y)

    def test_shape(self):
        # Test channels_last
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomCrop(16)(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = keras.KerasTensor((None, 3, None, None))
        y = RandomCrop(16)(x)
        self.assertEqual(y.shape, (None, 3, 16, 16))

        # Test static shape
        backend.set_image_data_format("channels_last")
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomCrop(16)(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

    def test_model(self):
        layer = RandomCrop(16)
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 16, 16, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedRandomCrop(16)
        y = layer(x)

        layer = FixedRandomCrop.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomCrop(16)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 16, 16, 3))

    def test_augment_bounding_box(self):
        # Test full bounding boxes
        images = np.zeros([1, 32, 32, 3]).astype("float32")
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
        layer = FixedRandomCrop((18, 18), bounding_box_format="rel_xyxy")

        output = layer(input)
        self.assertAllClose(output["bounding_boxes"]["boxes"], boxes["boxes"])
        self.assertAllClose(
            output["bounding_boxes"]["classes"], boxes["classes"]
        )

        # Test arbitrary bounding boxes
        images = np.zeros([1, 32, 32, 3]).astype("float32")
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
        layer = FixedRandomCrop((18, 18), bounding_box_format="xyxy")

        output = layer(input)
        expected_boxes = {
            "boxes": np.array(
                [
                    [[0, 0, 7, 10], [0, 4, 6, 11]],
                    [[0, 0, 0, 0], [5, 4, 7, 12]],
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
        ref_masks = masks[:, 8 : 8 + 16, 10 : 10 + 16, :]
        layer = FixedRandomCrop(16)
        output = layer(inputs)
        self.assertAllClose(output["segmentation_masks"], ref_masks)
