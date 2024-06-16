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
