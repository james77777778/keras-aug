try:
    import tensorflow_datasets
except ImportError:
    tensorflow_datasets = None


def assert_tfds_installed(symbol_name):
    if tensorflow_datasets is None:
        raise ImportError(
            f"{symbol_name} requires the `tensorflow_datasets` package. "
            "Please install the package using "
            "`pip install tensorflow_datasets`."
        )
