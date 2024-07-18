import keras
import numpy as np
import pytest
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.max_bounding_box import MaxBoundingBox
from keras_aug._src.utils.test_utils import get_images


class MaxBoundingBoxTest(testing.TestCase, parameterized.TestCase):
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
        inputs = {
            "images": get_images(dtype, "channels_last"),
            "bounding_boxes": {
                "boxes": np.ones((2, 4, 4)),
                "classes": np.ones((2, 4)),
            },
        }
        layer = MaxBoundingBox(max_number=8, dtype=dtype)
        outputs = layer(inputs)
        self.assertAllClose(
            outputs["bounding_boxes"]["boxes"][:, :4, :],
            inputs["bounding_boxes"]["boxes"],
        )
        self.assertAllClose(
            outputs["bounding_boxes"]["classes"][:, :4],
            inputs["bounding_boxes"]["classes"],
        )
        self.assertAllClose(
            outputs["bounding_boxes"]["boxes"][:, 4:, :],
            np.ones((2, 4, 4)) * -1,
        )
        self.assertAllClose(
            outputs["bounding_boxes"]["classes"][:, 4:],
            np.ones((2, 4)) * -1,
        )

    def test_shape(self):
        # Test static shape
        x = {
            "images": keras.KerasTensor((None, 32, 32, 3)),
            "bounding_boxes": {
                "boxes": keras.KerasTensor((None, 4, 4)),
                "classes": keras.KerasTensor((None, 4)),
            },
        }
        y = MaxBoundingBox(max_number=8)(x)
        self.assertEqual(y["bounding_boxes"]["boxes"].shape, (None, 8, 4))
        self.assertEqual(y["bounding_boxes"]["classes"].shape, (None, 8))

    @pytest.mark.skip("keras.models.Model doesn't support nested inputs")
    def test_model(self):
        layer = MaxBoundingBox(max_number=8)
        inputs = {
            "images": keras.KerasTensor((None, 32, 32, 3)),
            "bounding_boxes": {
                "boxes": keras.KerasTensor((None, 4, 4)),
                "classes": keras.KerasTensor((None, 4)),
            },
        }
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(
            model.output_shape["bounding_boxes"]["boxes"].shape, (None, 8, 4)
        )
        self.assertEqual(
            model.output_shape["bounding_boxes"]["classes"].shape, (None, 8)
        )

    def test_config(self):
        x = get_images("float32", "channels_last")
        inputs = {
            "images": x,
            "bounding_boxes": {
                "boxes": np.ones((2, 4, 4)),
                "classes": np.ones((2, 4)),
            },
        }
        layer = MaxBoundingBox(max_number=8)
        outputs = layer(inputs)
        boxes = keras.ops.convert_to_numpy(outputs["bounding_boxes"]["boxes"])
        classes = keras.ops.convert_to_numpy(
            outputs["bounding_boxes"]["classes"]
        )

        layer = MaxBoundingBox.from_config(layer.get_config())
        outputs2 = layer(inputs)
        boxes2 = keras.ops.convert_to_numpy(outputs2["bounding_boxes"]["boxes"])
        classes2 = keras.ops.convert_to_numpy(
            outputs2["bounding_boxes"]["classes"]
        )

        self.assertAllClose(boxes, boxes2)
        self.assertAllClose(classes, classes2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = MaxBoundingBox(max_number=8)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.map(
            lambda x: {
                "images": x,
                "bounding_boxes": {
                    "boxes": np.ones((4, 4)),
                    "classes": np.ones((4)),
                },
            }
        )
        ds = ds.batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output["images"], tf.Tensor)
            self.assertEqual(output["images"].shape, (2, 32, 32, 3))
            self.assertEqual(output["bounding_boxes"]["boxes"].shape, (2, 8, 4))
            self.assertEqual(output["bounding_boxes"]["classes"].shape, (2, 8))
