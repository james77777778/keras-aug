import pytest
import tensorflow as tf


def pytest_configure():
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


def pytest_addoption(parser):
    parser.addoption(
        "--skip-large",
        action="store_true",
        default=False,
        help="skip large tests",
    )


def pytest_collection_modifyitems(config, items):
    skip_large_tests = config.getoption("--skip-large")
    skip_large = pytest.mark.skipif(
        skip_large_tests, reason="need no --skip-large option to run"
    )
    for item in items:
        if "large" in item.keywords:
            item.add_marker(skip_large)
