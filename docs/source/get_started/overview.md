<!-- markdownlint-disable MD033 -->
# Overview

## Introduction

KerasAug is a library that includes pure TF/Keras preprocessing and augmentation layers, providing support for various data types such as images, labels, bounding boxes, segmentation masks, and more.

<div align="center">
<img style="width: 300px; max-width: 45%;" src="https://user-images.githubusercontent.com/20734616/238527700-7c81210f-f2ea-45a0-b4a7-bb4f6b936ec2.gif"> <img style="width: 300px; max-width: 45%;" src="https://user-images.githubusercontent.com/20734616/238522651-bdc48b92-4337-40b7-a911-f385d54324c1.gif">
</div>

> **Note**
> left: the visualization of the layers in KerasAug; right: the visualization of the YOLOV8 pipeline using KerasAug

KerasAug aims to provide fast, robust and user-friendly preprocessing and augmentation layers, facilitating seamless integration with TensorFlow, Keras and KerasCV.

KerasAug is:

- faster than KerasCV which is official Keras library
- supporting various data types, including **images, labels, bounding boxes, segmentation masks**, and more.
- dependent only on TensorFlow, TensorFlow Probability and KerasCV
- seamlessly integrating with the `tf.data` and `tf.keras.Model` API
- compatible with GPU

## Why KerasAug?

1. KerasAug is generally faster than KerasCV

    > RandomCropAndResize in KerasAug exhibits a remarkable speed-up of **+1342** compared to KerasAug. See [keras-aug/benchmarks](https://github.com/james77777778/keras-aug/tree/main/benchmarks) for more details.

2. The APIs of KerasAug are highly stable compared to KerasCV

    > KerasCV struggles to reproduce the YOLOV8 training pipeline, whereas KerasAug executes it flawlessly. See [Quickstart](https://github.com/james77777778/keras-aug/tree/main#quickstart) for more details.

3. KerasAug offers the functionality of sanitizing bounding boxes, ensuring the validity

    > The current layers in KerasAug support the sanitizing process by incorporating the `bounding_box_min_area_ratio` and `bounding_box_max_aspect_ratio` arguments.
    > In addition, you can bring the sanitizing functionality to your custom layer by utilizing `keras_aug.utils.bounding_box.sanitize_bounding_boxes`.

    <div align="center"><img style="max-width: 75%;" src="https://user-images.githubusercontent.com/20734616/238520600-34f0b7b5-d9ee-4483-859a-51e9644ded4c.jpg"></div>

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
