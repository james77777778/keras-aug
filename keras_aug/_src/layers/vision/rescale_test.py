import keras
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.rescale import Rescale
from keras_aug._src.utils.test_utils import get_images


class RescaleTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    @parameterized.named_parameters(
        named_product(dtype=["float32", "bfloat16"])
    )
    def test_correctness(self, dtype):
        if dtype == "bfloat16":
            atol = 1e-2
        else:
            atol = 1e-6

        x = get_images(dtype, "channels_last")
        layer = Rescale(scale=0.5, offset=0.1, dtype=dtype)
        y = layer(x)

        ref_y = x * 0.5 + 0.1
        self.assertDType(y, dtype)
        self.assertAllClose(y, ref_y, atol=atol)

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = Rescale(scale=2, offset=0.5)(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = Rescale(scale=2, offset=0.5)(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = Rescale(scale=2, offset=0.5)
        inputs = keras.layers.Input(shape=[None, None, 5])
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 5))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = Rescale(scale=2, offset=0.5)
        y = layer(x)

        layer = Rescale.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = Rescale(scale=2, offset=0.5)
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))
