---
title: KerasAug
app_file: app.py
sdk: gradio
sdk_version: 4.37.2
---

# KerasAug

<!-- markdownlint-disable MD033 -->

![Python](https://img.shields.io/badge/python-v3.9.0+-success.svg)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/james77777778/keras-aug/actions.yml?label=tests)](https://github.com/james77777778/keras-aug/actions/workflows/actions.yml?query=branch%3Amain++)
[![codecov](https://codecov.io/gh/james77777778/keras-aug/branch/main/graph/badge.svg?token=81ELI3VH7H)](https://codecov.io/gh/james77777778/keras-aug)
[![PyPI](https://img.shields.io/pypi/v/keras-aug)](https://pypi.org/project/keras-aug/)
![PyPI - Downloads](https://img.shields.io/pypi/dm/keras-aug)
[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/james77777778/KerasAug)

## Description

KerasAug is a library that includes Keras 3 preprocessing and augmentation layers, providing support for various data types such as images, labels, bounding boxes, segmentation masks, and more.

<div align="center">
<img width="45%" src="https://github.com/user-attachments/assets/bf9488c4-5c6b-4c87-8fa8-30170a67c92c" alt="object_detection.gif"> <img width="45%" src="https://github.com/user-attachments/assets/556db949-9461-438a-b1cf-3621ec63416e"  alt="semantic_segmentation.gif">
</div>
> **Note**
> Left: [YOLOV8 object detection pipeline](guides/voc_yolov8_aug.py), Right: [YOLOV8 semantic segmentation pipeline](guides/oxford_yolov8_aug.py)

KerasAug aims to provide fast, robust and user-friendly preprocessing and augmentation layers, facilitating seamless integration with Keras 3 and `tf.data.Dataset`.

The APIs largely follow `torchvision`, and the correctness of the layers has been verified through unit tests.

Also, you can check out the demo app on HF:

[![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/james77777778/KerasAug)

## Installation

```bash
pip install keras keras-aug -U
```

## Quickstart

### Rock, Paper and Scissors Image Classification

```python
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_aug import layers as ka_layers

BATCH_SIZE = 64
NUM_CLASSES = 3
INPUT_SIZE = (128, 128)

# Create a `tf.data.Dataset`-compatible preprocessing pipeline with all backends
train_dataset, validation_dataset = tfds.load(
    "rock_paper_scissors", as_supervised=True, split=["train", "test"]
)
train_dataset = (
    train_dataset.batch(BATCH_SIZE)
    .map(
        lambda images, labels: {
            "images": tf.cast(images, "float32") / 255.0,
            "labels": tf.one_hot(labels, NUM_CLASSES),
        }
    )
    .map(ka_layers.vision.Resize(INPUT_SIZE))
    .shuffle(128)
    .map(ka_layers.vision.RandAugment())
    .map(ka_layers.vision.CutMix(num_classes=NUM_CLASSES))
    .map(lambda data: (data["images"], data["labels"]))
    .prefetch(tf.data.AUTOTUNE)
)
validation_dataset = (
    validation_dataset.batch(BATCH_SIZE)
    .map(
        lambda images, labels: {
            "images": tf.cast(images, "float32") / 255.0,
            "labels": tf.one_hot(labels, NUM_CLASSES),
        }
    )
    .map(ka_layers.vision.Resize(INPUT_SIZE))
    .map(lambda data: (data["images"], data["labels"]))
    .prefetch(tf.data.AUTOTUNE)
)

# Create a CNN model
model = keras.models.Sequential(
    [
        keras.Input((*INPUT_SIZE, 3)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(256, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation="relu"),
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.AdamW(),
    metrics=["accuracy"],
)

# Train your model
model.fit(train_dataset, validation_data=validation_dataset, epochs=8)
```

The above example runs with all backends (JAX, TensorFlow, Torch).

### More Examples

- [YOLOV8 object detection pipeline](guides/voc_yolov8_aug.py)
- [YOLOV8 semantic segmentation pipeline](guides/oxford_yolov8_aug.py)

## Gradio App

```bash
gradio deploy
```

## Citing KerasAug

```bibtex
@misc{chiu2023kerasaug,
  title={KerasAug},
  author={Hongyu, Chiu},
  year={2023},
  howpublished={\url{https://github.com/james77777778/keras-aug}},
}
```
