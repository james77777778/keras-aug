# KerasAug

![Python](https://img.shields.io/badge/python-v3.8.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.12.0+-success.svg)
![Tensorflow Probability](https://img.shields.io/badge/tensorflow_probability-v0.19.0+-success.svg)
![KerasCV](https://img.shields.io/badge/keras_cv-v0.4.3+-success.svg)
[![Tests Status](https://github.com/james77777778/keras-aug/actions/workflows/actions.yml/badge.svg?branch=main)](https://github.com/james77777778/keras-aug/actions?query=branch%3Amain)
[![codecov](https://codecov.io/gh/james77777778/keras-aug/branch/main/graph/badge.svg?token=81ELI3VH7H)](https://codecov.io/gh/james77777778/keras-aug)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/james77777778/keras-aug/issues)

## Description

## Installation

KerasAug is compatible with the latest version of KerasCV, but is NOT compatible with `keras-cv < 0.5.0`.

```bash
pip install "keras-cv>=0.5.0" tensorflow tensorflow_probability --upgrade
```

## Usage

WIP

## Benchmark

Please refer to [benchmarks/README.md](benchmarks/README.md) for more details.

KerasAug is generally faster than KerasCV.

Unit: FPS (frames per second)

| Type           | Layer                   | KerasAug | KerasCV   |
|----------------|-------------------------|----------|-----------|
| Geometry       | RandomHFlip             | 2325     | 1769      |
|                | RandomVFlip             | 2012     | 1923      |
|                | RandomRotate            | 1896     | 1782      |
|                | RandomAffine            | 1901     | 818       |
|                | RandomCropAndResize     | 2480     | 210       |
|                | Resize (224, 224)       | 2550     | 213       |
| Intensity      | RandomBrightness        | 3054     | 2925      |
|                | RandomContrast          | 2941     | 3086      |
|                | RandomBrighnessContrast | 3009     | 629       |
|                | RandomColorJitter       | 2201     | 1120      |
|                | RandomGaussianBlur      | 2632     | 196       |
|                | Invert                  | 2933     | X         |
|                | Grayscale               | 3072     | 2762      |
|                | Equalize                | 204      | 140       |
|                | AutoContrast            | 2873     | 2744      |
|                | Posterize               | 3081     | 2929      |
|                | Solarize                | 2828     | 2560      |
|                | Sharpness               | 2554     | 2560      |
| Regularization | RandomCutout            | 2995     | 2978      |
|                | RandomGridMask          | 904      | 202       |
| Mix            | CutMix                  | 2352     | 2780      |
|                | MixUp                   | 2596     | 2962      |
| Auto           | AugMix                  | 80       | X (Error) |
|                | RandAugment             | 283      | 253       |
