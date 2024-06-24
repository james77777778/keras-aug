import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_resized_crop import RandomResizedCrop
from keras_aug._src.utils.test_utils import get_images


class FixedRandomResizedCrop(RandomResizedCrop):
    def get_params(self, batch_size, images=None, **kwargs):
        return dict(top=10, left=5, height=8, width=16)


class RandomResizedCropTest(testing.TestCase, parameterized.TestCase):
    pil_modes_mapping = {"nearest": 0, "bilinear": 2, "bicubic": 3}

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
            size=[(32, 32), (40, 50), (64, 64)],
            interpolation=["nearest", "bilinear", "bicubic"],
            antialias=[True, False],
            dtype=["float32", "uint8"],
        )
    )
    def test_correctness(self, size, interpolation, antialias, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        if size == (40, 50) and interpolation == "nearest":
            self.skipTest("TODO: Need to investigate")
        if interpolation == "nearest" and antialias is True:
            self.skipTest("Doesn't support nearest and antialias=True")
        if interpolation == "bicubic":
            self.skipTest("TODO: Need to investigate")
        torch_interpolation = self.pil_modes_mapping[interpolation]

        if dtype == "uint8":
            atol = 1.0
            rtol = 1e-6
            if interpolation == "bicubic":
                atol = 36.0
                rtol = 15.0
        else:
            atol = 1e-1
            rtol = 1e-6

        # Test channels_last
        np.random.seed(42)
        x = get_images(dtype, "channels_last")
        layer = FixedRandomResizedCrop(
            size, interpolation=interpolation, antialias=antialias, dtype=dtype
        )
        y = layer(x)

        ref_y = TF.resized_crop(
            torch.tensor(np.transpose(x.copy(), [0, 3, 1, 2])),
            10,
            5,
            8,
            16,
            size,
            torch_interpolation,
            antialias,
        )
        ref_y = np.transpose(ref_y.cpu().numpy(), [0, 2, 3, 1])
        self.assertAllClose(y, ref_y, atol=atol, rtol=rtol)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        np.random.seed(42)
        x = get_images(dtype, "channels_first")
        layer = FixedRandomResizedCrop(
            size, interpolation=interpolation, antialias=antialias, dtype=dtype
        )
        y = layer(x)

        ref_y = TF.resized_crop(
            torch.tensor(x.copy()),
            10,
            5,
            8,
            16,
            size,
            torch_interpolation,
            antialias,
        )
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y, atol=atol, rtol=rtol)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomResizedCrop(16)(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomResizedCrop(16)(x)
        self.assertEqual(y.shape, (None, 16, 16, 3))

    def test_model(self):
        # Test dynamic shape
        layer = RandomResizedCrop(16)
        inputs = keras.layers.Input(shape=[None, None, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 16, 16, 3))

        # Test static shape
        layer = RandomResizedCrop((16, 32))
        inputs = keras.layers.Input(shape=[32, 32, 3])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 16, 32, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedRandomResizedCrop(16)
        y = layer(x)

        layer = FixedRandomResizedCrop.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomResizedCrop(16)
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
        input = {"images": images, "bounding_boxes": boxes.copy()}
        layer = FixedRandomResizedCrop((18, 18), bounding_box_format="rel_xyxy")

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
        input = {"images": images, "bounding_boxes": boxes.copy()}
        layer = FixedRandomResizedCrop((16, 32), bounding_box_format="xyxy")

        output = layer(input)
        expected_boxes = {
            "boxes": np.array(
                [
                    [[0, 0, 24, 16], [10, 4, 22, 16]],
                    [[0, 0, 0, 0], [20, 4, 24, 16]],
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
        ref_masks = masks[:, 10 : 10 + 8, 5 : 5 + 16, :]
        layer = FixedRandomResizedCrop((8, 16))
        output = layer(inputs)
        self.assertAllClose(output["segmentation_masks"], ref_masks)
