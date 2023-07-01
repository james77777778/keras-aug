try:
    import tensorflow_datasets
except ImportError:
    tensorflow_datasets = None
try:
    import keras_cv
except ImportError:
    keras_cv = None


def assert_tfds_installed(symbol_name):
    if tensorflow_datasets is None:
        raise ImportError(
            f"{symbol_name} requires the `tensorflow_datasets` package. "
            "Please install the package using "
            "`pip install tensorflow_datasets`."
        )


def assert_kerascv_installed(symbol_name):
    if keras_cv is None:
        raise ImportError(
            f"{symbol_name} requires the `keras_cv` package. "
            "Please install the package using `pip install keras_cv`."
        )
