import ml_dtypes as ml_dtypes
import numpy as np
from keras import distribution


def get_images(dtype, data_format="channels_first", size=(32, 32)):
    # channels_first
    if dtype == "float32":
        x = np.random.uniform(0, 1, (2, 3, *size)).astype(dtype)
    elif dtype == "bfloat16":
        x = np.random.uniform(0, 1, (2, 3, *size)).astype(dtype)
    elif dtype == "uint8":
        x = np.random.uniform(0, 255, (2, 3, *size)).astype(dtype)
    elif dtype == "int8":
        x = np.random.uniform(-128, 127, (2, 3, *size)).astype(dtype)
    if data_format == "channels_last":
        x = np.transpose(x, [0, 2, 3, 1])
    return x


def uses_gpu():
    # Condition used to skip tests when using the GPU
    devices = distribution.list_devices()
    if any(d.startswith("gpu") for d in devices):
        return True
    return False
