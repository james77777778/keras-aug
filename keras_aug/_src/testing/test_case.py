import numpy as np
from absl.testing import parameterized
from keras import backend
from keras import ops
from keras.src import testing


class TestCase(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def convert_to_numpy(self, inputs):
        import torch
        from keras.src.backend.torch import convert_to_numpy

        if isinstance(inputs, torch.Tensor):
            inputs = convert_to_numpy(inputs)
        if not isinstance(inputs, np.ndarray):
            inputs = ops.convert_to_numpy(inputs)
        return inputs

    def assertAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        x1 = self.convert_to_numpy(x1)
        x2 = self.convert_to_numpy(x2)
        if backend.standardize_dtype(x1.dtype) == "bfloat16":
            x1 = x1.astype("float32")
        if backend.standardize_dtype(x2.dtype) == "bfloat16":
            x2 = x2.astype("float32")
        super().assertAllClose(x1, x2, atol, rtol, msg)

    def assertNotAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        x1 = self.convert_to_numpy(x1)
        x2 = self.convert_to_numpy(x2)
        if backend.standardize_dtype(x1.dtype) == "bfloat16":
            x1 = x1.astype("float32")
        if backend.standardize_dtype(x2.dtype) == "bfloat16":
            x2 = x2.astype("float32")
        super().assertNotAllClose(x1, x2, atol, rtol, msg)

    def assertDType(self, x, dtype, msg=None):
        dtype = dtype.replace("mixed_", "")
        return super().assertDType(x, dtype, msg)
