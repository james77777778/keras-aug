import os

import pytest


def pytest_configure():
    import tensorflow as tf

    # disable tensorflow gpu memory preallocation
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # disable jax gpu memory preallocation
    # https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


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
