<!-- markdownlint-disable MD033 -->
# Overview

## Introduction

KerasAug is a library that includes pure TF/Keras preprocessing and augmentation layers, providing support for various data types such as images, labels, bounding boxes, segmentation masks, and more.

<div align="center"><img style="width: 440px; max-width: 90%;" src="https://user-images.githubusercontent.com/20734616/237416247-417f2870-1e0d-45d6-abda-e384a82118df.gif"></div>

KerasAug aims to provide fast and user-friendly preprocessing and augmentation layers, facilitating seamless integration with TensorFlow, Keras, and KerasCV.

KerasAug is:

- built entirely using TensorFlow, TensorFlow Probability, Keras and KerasCV
- supporting various data types, including images, labels, bounding boxes, segmentation masks, and more.
- compatible with GPU (partially compatible with TPU/XLA)
- seamlessly integrating with the `tf.data` and `tf.keras.Model` API
- cosistent with officially published implementations

## Data Format

KerasAug expects following types of input data:

1. `tf.Tensor` with the shape of [H, W, C] representing `"images"`
2. `tf.Tensor` or `tf.RaggedTensor` with the shape of [B, H|None, W|None, C] representing `"images"`
3. A dictionary containing:
    - `"images"`: [B, H|None, W|None, C] (`tf.Tensor` or `tf.RaggedTensor`)
    - `"labels"`: [B, 1] (`tf.Tensor`)
    - `"bounding_boxes"`:
        - `"boxes"`: [B, N|None, 4] (`tf.Tensor` or `tf.RaggedTensor`)
        - `"classes"`: [B, N|None, 1] `tf.Tensor` or `tf.RaggedTensor`
    - `"segmentation_masks"`: [B, H|None, W|None, 1] (`tf.Tensor` or `tf.RaggedTensor`)
    - `"keypoints"`: WIP
    - `"custom_annotations"`: user-defined

For example:

```python
# an image
images = tf.random.uniform((224, 224, 3)) * 255.0

# a batch of images
images = tf.random.uniform((4, 224, 224, 3)) * 255.0

# a batch of ragged images
images = tf.ragged.stack(
    [
        tf.ones((224, 224, 3)),
        tf.ones((320, 320, 3)),
    ]
)
print(isinstance(images, tf.RaggedTensor))  # True

# a batch of ragged images with ragged bounding_boxes
data = {
    "images": tf.ragged.stack(
        [
            tf.ones((224, 224, 3)),
            tf.ones((320, 320, 3)),
        ]
    ),
    "bounding_boxes": {
        "boxes": tf.ragged.constant(
            [
                [[100, 100, 200, 200], [50, 50, 150, 150]],  # 2 boxes in the first image
                [[200, 200, 300, 300]],  # 1 box in the second image
            ],
            dtype=tf.float32,
        ),
        "classes": tf.ragged.constant(
            [[0, 0], [0]],  # all boxes belong to 0 class
            dtype=tf.float32,
        ),
    }
}
```

Refer to [keras-aug/examples](https://github.com/james77777778/keras-aug/tree/main/examples) for practical use.
