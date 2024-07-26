import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.mosaic import Mosaic
from keras_aug._src.utils.test_utils import get_images


class FixedMosaic(Mosaic):
    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        return dict(
            centers_x=ops.numpy.ones([batch_size]) * 0.75,
            centers_y=ops.numpy.ones([batch_size]) * 0.75,
        )


class MosaicTest(testing.TestCase, parameterized.TestCase):
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
        # Test channels_last
        images = get_images(dtype, "channels_last")
        inputs = [np.rot90(images, k=i, axes=(1, 2)) for i in range(4)]
        layer = FixedMosaic(size=(32, 32), dtype=dtype)
        outputs = layer(*inputs)

        self.assertDType(outputs, dtype)
        self.assertAllClose(outputs, inputs[0])

        # Test channels_first
        backend.set_image_data_format("channels_first")
        images = get_images(dtype, "channels_first")
        inputs = [np.rot90(images, k=i, axes=(2, 3)) for i in range(4)]
        layer = FixedMosaic(size=(32, 32), dtype=dtype)
        outputs = layer(*inputs)

        self.assertDType(outputs, dtype)
        self.assertAllClose(outputs, inputs[0])

    def test_shape(self):
        # Test dynamic shape
        x = [keras.KerasTensor((None, None, None, 3)) for _ in range(4)]
        y = Mosaic(size=(32, 32))(*x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

        # Test static shape
        x = [keras.KerasTensor((None, 32, 32, 3)) for _ in range(4)]
        y = Mosaic(size=(32, 32))(*x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = Mosaic(size=(32, 32))
        inputs = [keras.layers.Input(shape=(None, None, 3)) for _ in range(4)]
        outputs = layer(*inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, 32, 32, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        x = [np.rot90(x, k=i, axes=(1, 2)) for i in range(4)]
        layer = FixedMosaic(size=(32, 32))
        y = layer(*x)

        layer = FixedMosaic.from_config(layer.get_config())
        y2 = layer(*x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = Mosaic(size=(32, 32))
        x = get_images("float32", "channels_last")

        def duplicate(x):
            return x, x, x, x

        ds = (
            tf.data.Dataset.from_tensor_slices(x)
            .map(duplicate)
            .batch(2)
            .map(layer)
        )
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))

    def test_augment_bounding_box(self):
        from keras_aug._src.layers.vision.random_rotation import RandomRotation

        # Test full bounding boxes
        images = np.zeros([1, 32, 32, 3]).astype("float32")
        boxes = {
            "boxes": np.array(
                [
                    [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
                ],
                "float32",
            ),
            "classes": np.array([[0, 0, 0, 0]], "float32"),
        }
        input = {"images": images, "bounding_boxes": boxes}
        input_r90 = RandomRotation([90, 90], bounding_box_format="rel_xyxy")(
            input
        )
        input_r180 = RandomRotation([180, 180], bounding_box_format="rel_xyxy")(
            input
        )
        input_r270 = RandomRotation([270, 270], bounding_box_format="rel_xyxy")(
            input
        )
        layer = FixedMosaic(size=(32, 32), bounding_box_format="rel_xyxy")

        output = layer(input, input_r90, input_r180, input_r270)
        self.assertAllClose(
            output["bounding_boxes"]["boxes"][0, :4], boxes["boxes"][0]
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"][0, :4], boxes["classes"][0]
        )

        # Test arbitrary bounding boxes
        images = np.zeros([1, 32, 32, 3]).astype("float32")
        boxes = {
            "boxes": np.array(
                [
                    [
                        [3, 4, 17, 18],
                        [10, 12, 16, 19],
                        [0, 0, 1, 1],
                        [15, 12, 17, 20],
                    ],
                ],
                "float32",
            ),
            "classes": np.array([[0, 1, 2, 3]], "float32"),
        }
        input = {"images": images, "bounding_boxes": boxes}
        layer = FixedMosaic(size=(32, 32), bounding_box_format="xyxy")

        output = layer(input, input, input, input)
        self.assertAllClose(
            output["bounding_boxes"]["boxes"][0, :4], boxes["boxes"][0]
        )
        self.assertAllClose(
            output["bounding_boxes"]["classes"][0, :4], boxes["classes"][0]
        )

    def test_augment_segmentation_mask(self):
        num_classes = 8
        images_shape = (1, 32, 32, 3)
        masks_shape = (1, 32, 32, 1)
        images = np.random.uniform(size=images_shape).astype("float32")
        masks = np.random.randint(2, size=masks_shape) * (num_classes - 1)
        inputs = {"images": images, "segmentation_masks": masks}

        # Crop to exactly 1/2 of the size
        layer = FixedMosaic(size=(32, 32))
        output = layer(inputs, inputs, inputs, inputs)
        self.assertAllClose(output["segmentation_masks"], masks)
