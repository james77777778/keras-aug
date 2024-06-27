import keras
import numpy as np
from absl.testing import parameterized
from keras import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.layers.vision.random_erasing import RandomErasing
from keras_aug._src.utils.test_utils import get_images


class FixedRandomErasing(RandomErasing):
    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        images_shape = ops.shape(images)
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]
        top = ops.numpy.zeros([batch_size])
        left = ops.numpy.zeros([batch_size])
        h = ops.numpy.ones([batch_size]) * 10
        w = ops.numpy.ones([batch_size]) * 10
        if isinstance(self.value, str):
            dtype = backend.result_type(images.dtype, float)
            v = ops.random.normal(ops.shape(images), dtype=dtype)
        elif isinstance(self.value, float):
            dtype = backend.standardize_dtype(images.dtype)
            v = ops.numpy.full(ops.shape(images), self.value)
            v = ops.cast(v, dtype)
        elif isinstance(self.value, tuple):
            dtype = backend.standardize_dtype(images.dtype)
            v = ops.convert_to_tensor(self.value)  # [c]
            v = ops.cast(v, dtype)
            if self.data_format == "channels_last":
                v = ops.numpy.expand_dims(v, axis=[0, 1, 2])
                v = ops.numpy.tile(v, [batch_size, height, width, 1])
            else:
                v = ops.numpy.expand_dims(v, axis=[0, -1, -2])
                v = ops.numpy.tile(v, [batch_size, 1, height, width])
        return dict(top=top, left=left, height=h, width=w, value=v)


class RandomErasingTest(testing.TestCase, parameterized.TestCase):
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
            value=[0.0, (1.0, 1.0, 1.0), "random"], dtype=["float32", "uint8"]
        )
    )
    def test_correctness(self, value, dtype):
        if dtype == "uint8" and value == "random":
            self.skipTest("value='random' doesn't support dtype='uint8'")

        # Test channels_last
        images = get_images(dtype, "channels_last")
        layer = FixedRandomErasing(value=value, dtype=dtype)
        outputs = layer(images)

        self.assertDType(outputs, dtype)
        if value == 0.0:
            self.assertAllClose(
                outputs[:, 0:10, 0:10, :],
                np.zeros_like(outputs[:, 0:10, 0:10, :]),
            )
        elif value == (1.0, 1.0, 1.0):
            self.assertAllClose(
                outputs[:, 0:10, 0:10, :],
                np.ones_like(outputs[:, 0:10, 0:10, :]),
            )
        else:
            pass
        self.assertAllClose(outputs[:, 10:, 10:, :], images[:, 10:, 10:, :])

        # Test channels_first
        backend.set_image_data_format("channels_first")
        images = get_images(dtype, "channels_first")
        layer = FixedRandomErasing(value=value, dtype=dtype)
        outputs = layer(images)

        self.assertDType(outputs, dtype)
        if value == 0.0:
            self.assertAllClose(
                outputs[:, :, 0:10, 0:10],
                np.zeros_like(outputs[:, :, 0:10, 0:10]),
            )
        elif value == (1.0, 1.0, 1.0):
            self.assertAllClose(
                outputs[:, :, 0:10, 0:10],
                np.ones_like(outputs[:, :, 0:10, 0:10]),
            )
        else:
            pass
        self.assertAllClose(outputs[:, :, 10:, 10:], images[:, :, 10:, 10:])

    def test_shape(self):
        # Test dynamic shape
        x = keras.KerasTensor((None, None, None, 3))
        y = RandomErasing()(x)
        self.assertEqual(y.shape, (None, None, None, 3))

        # Test static shape
        x = keras.KerasTensor((None, 32, 32, 3))
        y = RandomErasing()(x)
        self.assertEqual(y.shape, (None, 32, 32, 3))

    def test_model(self):
        layer = RandomErasing()
        inputs = keras.layers.Input(shape=(None, None, 3))
        outputs = layer(inputs)
        model = keras.models.Model(inputs, outputs)
        self.assertEqual(model.output_shape, (None, None, None, 3))

    def test_config(self):
        x = get_images("float32", "channels_last")
        layer = FixedRandomErasing()
        y = layer(x)

        layer = FixedRandomErasing.from_config(layer.get_config())
        y2 = layer(x)
        self.assertAllClose(y, y2)

    def test_tf_data_compatibility(self):
        import tensorflow as tf

        layer = RandomErasing()
        x = get_images("float32", "channels_last")
        ds = tf.data.Dataset.from_tensor_slices(x).batch(2).map(layer)
        for output in ds.take(1):
            self.assertIsInstance(output, tf.Tensor)
            self.assertEqual(output.shape, (2, 32, 32, 3))
