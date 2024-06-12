import numpy as np
from absl.testing import parameterized
from keras import backend
from keras import ops
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.backend.image import ImageBackend


class ImageBackendTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_crop(self):
        image_backend = ImageBackend()

        # Test channels_last
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        y = image_backend.crop(x, top=5, left=6, height=13, width=14)
        self.assertAllClose(y, x[:, 5 : 5 + 13, 6 : 6 + 14, :])

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        y = image_backend.crop(x, top=5, left=6, height=13, width=14)
        self.assertAllClose(y, x[:, :, 5 : 5 + 13, 6 : 6 + 14])

    @parameterized.named_parameters(
        named_product(mode=["constant", "reflect", "symmetric"])
    )
    def test_pad(self, mode):
        image_backend = ImageBackend()

        # Test channels_last
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        y = image_backend.pad(x, mode, 2, 3, 4, 5)

        ref_y = ops.pad(
            x,
            [[0, 0], [2, 3], [4, 5], [0, 0]],
            mode,
            0 if mode == "constant" else None,
        )
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(0, 1, (2, 3, 32, 32)).astype("float32")
        y = image_backend.pad(x, mode, 2, 3, 4, 5)

        ref_y = ops.pad(
            x,
            [[0, 0], [0, 0], [2, 3], [4, 5]],
            mode,
            0 if mode == "constant" else None,
        )
        self.assertAllClose(y, ref_y)

    @parameterized.named_parameters(
        named_product(
            value_range=[(0.0, 1.0), (-0.5, 1.5), (1.0, 1.0), (0.0, 0.0)]
        )
    )
    def test_rgb_to_hsv(self, value_range):
        import tensorflow as tf

        image_backend = ImageBackend()
        low, high = value_range

        # Test channels_last
        x = np.random.uniform(low, high, (2, 32, 32, 3)).astype("float32")
        y = image_backend.rgb_to_hsv(x)

        ref_y = tf.image.rgb_to_hsv(x)
        self.assertAllClose(y, ref_y, atol=1e-3)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(low, high, (2, 3, 32, 32)).astype("float32")
        y = image_backend.rgb_to_hsv(x)

        ref_y = tf.image.rgb_to_hsv(tf.transpose(x, [0, 2, 3, 1]))
        ref_y = tf.transpose(ref_y, [0, 3, 1, 2])
        self.assertAllClose(y, ref_y, atol=1e-3)

    @parameterized.named_parameters(
        named_product(
            value_range=[(0.0, 1.0), (-0.5, 1.5), (1.0, 1.0), (0.0, 0.0)]
        )
    )
    def test_hsv_to_rgb(self, value_range):
        import tensorflow as tf

        image_backend = ImageBackend()
        low, high = value_range

        # Test channels_last
        x = np.random.uniform(low, high, (2, 32, 32, 3)).astype("float32")
        x = np.clip(x, 0.0, 1.0)
        y = image_backend.hsv_to_rgb(x)

        ref_y = tf.image.hsv_to_rgb(x)
        self.assertAllClose(y, ref_y, atol=1e-3)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = np.random.uniform(low, high, (2, 3, 32, 32)).astype("float32")
        x = np.clip(x, 0.0, 1.0)
        y = image_backend.hsv_to_rgb(x)

        ref_y = tf.image.hsv_to_rgb(tf.transpose(x, [0, 2, 3, 1]))
        ref_y = tf.transpose(ref_y, [0, 3, 1, 2])
        self.assertAllClose(y, ref_y, atol=1e-3)

    def test_rgb_to_hsv_roundtrip(self):
        image_backend = ImageBackend()

        # RGB -> HSV -> RGB
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        y = image_backend.rgb_to_hsv(x)
        y = image_backend.hsv_to_rgb(y)

        self.assertAllClose(y, x, atol=1e-3)

        # HSV -> RGB -> HSV
        x = np.random.uniform(0, 1, (2, 32, 32, 3)).astype("float32")
        y = image_backend.hsv_to_rgb(x)
        y = image_backend.rgb_to_hsv(y)

        self.assertAllClose(y, x, atol=1e-3)
