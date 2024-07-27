# KerasAug

<!-- markdownlint-disable MD033 -->

![Keras](https://img.shields.io/badge/keras-v3.4.1+-success.svg)
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

> [!NOTE]
> See `docs/*.py` for the GIF generation. YOLOV8-like pipeline for bounding boxes and segmentation masks.

KerasAug aims to provide fast, robust and user-friendly preprocessing and augmentation layers, facilitating seamless integration with Keras 3 and `tf.data`.

The APIs largely follow `torchvision`, and the correctness of the layers has been verified through unit tests.

Also, you can check out the demo app on HF:
App here: [![Open in HF Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/james77777778/KerasAug)

## Why KerasAug

- 🚀 Supports many preprocessing & augmentation layers across all backends (JAX, TensorFlow and Torch).
- 🧰 Seamlessly integrates with `tf.data`, offering a performant and scalable data pipeline.
- 🔥 Follows the same API design as `torchvision`.
- 🙌 Depends only on Keras 3.

## Installation

```bash
pip install keras keras-aug -U
```

> [!IMPORTANT]  
> Make sure you have installed a supported backend for Keras.

## Quickstart

### Rock, Paper and Scissors Image Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11xc0nW06iWQ_R-oH4wLB_MYV4GY4mNwy?usp=sharing)

```python
import keras
import tensorflow as tf
import tensorflow_datasets as tfds

from keras_aug import layers as ka_layers

BATCH_SIZE = 64
NUM_CLASSES = 3
INPUT_SIZE = (128, 128)

# Create a `tf.data.Dataset`-compatible preprocessing pipeline.
# Note that this example works with all backends.
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
    .map(ka_layers.vision.Rescale(scale=2.0, offset=-1))  # [0, 1] to [-1, 1]
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
    .map(ka_layers.vision.Rescale(scale=2.0, offset=-1))  # [0, 1] to [-1, 1]
    .map(lambda data: (data["images"], data["labels"]))
    .prefetch(tf.data.AUTOTUNE)
)

# Create a model using MobileNetV2 as the backbone.
backbone = keras.applications.MobileNetV2(
    input_shape=(*INPUT_SIZE, 3), include_top=False
)
backbone.trainable = False
inputs = keras.Input((*INPUT_SIZE, 3))
x = backbone(inputs)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9),
    metrics=["accuracy"],
)

# Train and evaluate your model
model.fit(train_dataset, validation_data=validation_dataset, epochs=8)
model.evaluate(validation_dataset)
```

The above example runs with all backends (JAX, TensorFlow, Torch).

### More Examples

- [YOLOV8 object detection pipeline](guides/voc_yolov8_aug.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AgnnvfTRMHKq--7gvmHP7RyxTeQResV4?usp=sharing)

- [YOLOV8 semantic segmentation pipeline](guides/oxford_yolov8_aug.py) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IJwUPiHreO7iIJ3VewgfLRoBdFdcQcJE?usp=sharing)

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
