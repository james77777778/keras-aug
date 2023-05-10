# KerasAug

![Python](https://img.shields.io/badge/python-v3.8.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.12.0+-success.svg)
![Tensorflow Probability](https://img.shields.io/badge/tensorflow_probability-v0.20.0+-success.svg)
![KerasCV](https://img.shields.io/badge/keras_cv-v0.5.0+-success.svg)
[![Tests Status](https://github.com/james77777778/keras-aug/actions/workflows/actions.yml/badge.svg?branch=main)](https://github.com/james77777778/keras-aug/actions?query=branch%3Amain)
[![codecov](https://codecov.io/gh/james77777778/keras-aug/branch/main/graph/badge.svg?token=81ELI3VH7H)](https://codecov.io/gh/james77777778/keras-aug)
[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/james77777778/keras-aug/issues)

## Description

KerasAug is a library that includes pure TF/Keras preprocessing and augmentation layers, providing support for various data types such as images, bounding boxes, segmentation masks, and more.

![visualization](https://user-images.githubusercontent.com/20734616/237416247-417f2870-1e0d-45d6-abda-e384a82118df.gif)

KerasAug aims to provide fast and user-friendly preprocessing and augmentation layers, facilitating seamless integration with TensorFlow, Keras, and KerasCV.

KerasAug is:

- built entirely using TensorFlow, TensorFlow Probability, Keras and KerasCV
- supporting various data types, including images, bounding boxes, segmentation masks, and more.
- compatible with GPU (partially compatible with TPU/XLA)
- seamlessly integrating with the `tf.data` and `tf.keras.Model` API
- cosistent with officially published implementations

## Installation

KerasAug is compatible with the latest version of KerasCV, but is NOT compatible with `keras-cv < 0.5.0`.

```bash
pip install "keras-cv>=0.5.0" tensorflow tensorflow_probability --upgrade
```

## Quick Start

```python
import keras_aug
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# Create a preprocessing pipeline with KerasAug
BATCH_SIZE = 16
NUM_CLASSES = 3
augmenter = keras.Sequential(
    [
        keras_aug.layers.RandomFlip(),
        keras_aug.layers.RandAugment(
            value_range=(0, 255), augmentations_per_image=3
        ),
        keras_aug.layers.CutMix(),
    ]
)


def preprocess_data(images, labels, augment=False):
    labels = tf.one_hot(labels, NUM_CLASSES)
    inputs = {"images": images, "labels": labels}
    outputs = augmenter(inputs) if augment else inputs
    return outputs['images'], outputs['labels']


train_dataset, test_dataset = tfds.load(
    'rock_paper_scissors', as_supervised=True, split=['train', 'test']
)
train_dataset = (
    train_dataset.batch(BATCH_SIZE)
    .map(
        lambda x, y: preprocess_data(x, y, augment=True),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    test_dataset.batch(BATCH_SIZE)
    .map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# Create a model using a pretrained backbone
backbone = keras_cv.models.ResNet50V2Backbone.from_preset(
    "resnet50_v2_imagenet"
)
model = keras_cv.models.ImageClassifier(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    activation="softmax",
)
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy'],
)

# Train your model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=8,
)
```

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
