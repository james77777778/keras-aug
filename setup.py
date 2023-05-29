"""Setup script. Copied and modified from
https://github.com/keras-team/keras-cv/blob/master/setup.py
"""

import pathlib

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution

BUILD_WITH_CUSTOM_OPS = False

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
DESCRIPTION = (
    "A library that includes pure TF/Keras preprocessing and augmentation "
    "layers, providing support for various data types such as images, labels, "
    "bounding boxes, segmentation masks, and more."
)


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return BUILD_WITH_CUSTOM_OPS

    def is_pure(self):
        return not BUILD_WITH_CUSTOM_OPS


setup(
    name="keras-aug",
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/james77777778/keras-aug",
    author="Hongyu, Chiu (james77777778)",
    author_email="",
    license="Apache License 2.0",
    install_requires=["packaging", "absl-py", "regex", "tensorflow-datasets"],
    python_requires=">=3.8",
    extras_require={
        "tests": [
            "ruff",
            "black[jupyter]",
            "pytest",
            "pytest-cov",
            "pycocotools",
            "tensorflow",
            "keras-cv",
        ],
        "examples": ["tensorflow-datasets", "matplotlib"],
    },
    distclass=BinaryDistribution,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    packages=find_packages(),
    include_package_data=True,
)
