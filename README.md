# KerasAug

![Python](https://img.shields.io/badge/python-v3.8.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.12.0+-success.svg)
![Tensorflow Probability](https://img.shields.io/badge/tensorflow_probability-v0.19.0+-success.svg)
[![Tests Status](https://github.com/james77777778/keras-aug/actions/workflows/actions.yml/badge.svg?branch=main)](https://github.com/james77777778/keras-aug/actions?query=branch%3Amain)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/james77777778/keras-aug/issues)

## Description

## Installation

Please follow the installation instructions in KerasCV:

```bash
pip install git+https://github.com/keras-team/keras-cv.git tensorflow tensorflow_probability --upgrade
```

## Usage

## Benchmark

See [benchmarks/README.md](benchmarks/README.md) for details.

Unit: FPS (frame per second)

||KerasAug|KerasCV|
|-|-|-|
|HorizontalFlip|2325|1769|
|VerticalFlip|2012|1923|
|RandomRotate|1896|1782|
|RandomAffine|1901|818|
|RandomBrightness|3054|2925|
|RandomContrast\*|2941|3086|
|RandomBrighnessContrast\*|3009|629|
|RandomColorJitter\*|2201|1120|
|Grayscale|3072|2762|
|Resize (224, 224)|2593|210|
|RandomCropAndResize|2480|210|
|Equalize|204|140|
|RandomGaussianBlur|2632|196|
|CutMix|2352|2780|
|MixUp|2596|2962|
|AutoContrast|2873|2744|
|Posterize|3081|2929|
|Solarize|2828|2560|
|Sharpness|2554|2560|
|Invert|2933|X|
