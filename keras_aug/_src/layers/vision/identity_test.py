import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.identity import Identity
from keras_aug._src.utils.test_utils import get_images


class IdentityTest(testing.TestCase, parameterized.TestCase):
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
        x = get_images(dtype, "channels_last")
        layer = Identity(dtype=dtype)
        y = layer(x)
        self.assertAllClose(y, x)

        x = {
            "images": get_images(dtype, "channels_last"),
            "bounding_boxes": {
                "boxes": np.random.uniform(0, 1, (2, 10, 4)),
                "classes": np.random.uniform(0, 1, (2, 10, 5)),
            },
            "segmentation_masks": np.random.uniform(
                0, 9, (2, 32, 32, 1)
            ).astype("int32"),
            "keypoints": np.random.uniform(0, 1, (2, 10, 17)),
        }
        y = layer(x)
        self.assertDType(y["images"], dtype)
        self.assertAllClose(y["images"], x["images"])
        self.assertAllClose(
            y["bounding_boxes"]["boxes"], x["bounding_boxes"]["boxes"]
        )
        self.assertAllClose(
            y["bounding_boxes"]["classes"], x["bounding_boxes"]["classes"]
        )
        self.assertAllClose(y["segmentation_masks"], x["segmentation_masks"])
        self.assertAllClose(y["keypoints"], x["keypoints"])

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = Identity()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = Identity()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = Identity()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = Identity()
        y = layer(x)

        layer = Identity.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = Identity()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))
