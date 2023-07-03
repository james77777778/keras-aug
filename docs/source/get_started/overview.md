<!-- markdownlint-disable MD033 -->
# Overview

## Introduction

KerasAug is a library that includes pure TF/Keras preprocessing and augmentation layers, providing support for various data types such as images, labels, bounding boxes, segmentation masks, and more.

<div align="center">
<img width="45%" src="https://user-images.githubusercontent.com/20734616/238531125-f0a07f50-423b-4be2-9dcd-a3cc459a261c.gif"> <img width="45%" src="https://user-images.githubusercontent.com/20734616/238531295-22cd5567-0709-46d5-bf31-7baad05b91d2.gif">
</div>

> **Note**
> left: the visualization of the layers in KerasAug; right: the visualization of the YOLOV8 pipeline using KerasAug

KerasAug aims to provide fast, robust and user-friendly preprocessing and augmentation layers, facilitating seamless integration with TensorFlow, Keras and KerasCV.

KerasAug is:

- 🚀 faster than [KerasCV](https://github.com/keras-team/keras-cv) which is an official Keras library
- 🧰 supporting various data types, including **images, labels, bounding boxes, segmentation masks**, and more.
- ❤️ dependent only on TensorFlow
- 🌟 seamlessly integrating with the `tf.data` and `tf.keras.Model` APIs
- 🔥 compatible with GPU and mixed precision (`mixed_float16` and `mixed_bfloat16`)

Check out the demo website powered by Streamlit:

<a href="https://james77777778-keras-aug-streamlit-appapp-mxd7v1.streamlit.app/"><img width="50%" align="right" src="https://user-images.githubusercontent.com/20734616/242836830-bd0a457d-fa6f-410c-a267-af628f5bb5ec.JPG"></a>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://james77777778-keras-aug-streamlit-appapp-mxd7v1.streamlit.app/)

- Apply a transformation to the default or uploaded image
- Adjust the arguments of the specified layer

## Why KerasAug?

1. KerasAug is generally faster than KerasCV

    > RandomCropAndResize in KerasAug exhibits a remarkable speed-up of **+1150%** compared to KerasAug. See [keras-aug/benchmarks](https://github.com/james77777778/keras-aug/tree/main/benchmarks) for more details.

2. The APIs of KerasAug are highly stable compared to KerasCV

    > KerasCV struggles to reproduce the YOLOV8 training pipeline, whereas KerasAug executes it flawlessly. See [Quickstart](https://github.com/james77777778/keras-aug/tree/main#quickstart) for more details.

3. KerasAug comes with built-in support for mixed precision training

    > All layers in KerasAug can run with `tf.keras.mixed_precision.set_global_policy('mixed_float16')`

4. KerasAug offers the functionality of sanitizing bounding boxes, ensuring the validity

    > The current layers in KerasAug support the sanitizing process by incorporating the `bounding_box_min_area_ratio` and `bounding_box_max_aspect_ratio` arguments.

    <div align="center"><img width="60%" src="https://user-images.githubusercontent.com/20734616/238520600-34f0b7b5-d9ee-4483-859a-51e9644ded4c.jpg"></div>

    > **Note**
    > The degenerate bounding boxes (those located at the bottom of the image) are removed.

## Data Format

KerasAug expects following types of input data:

||Type|Key|Shape|Notes|
|-|-|-|-|-|
|Single Image|`tf.Tensor`||[H, W, C]|
|Multiple Images|`tf.Tensor`||[B, H, W, C]|H, W can be `None` if ragged|
|Multiple Inputs|`dict`|`images`|[B, H, W, C]|H, W can be `None` if ragged|
|||`labels`|[B, 1]||
|||`bounding_boxes`<br>(boxes)|[B, N, 4]|N can be `None` if ragged|
|||`bounding_boxes`<br>(classes)|[B, N]|N can be `None` if ragged|
|||`segmentation_masks`|[B, H, W, 1]<br>or [B, H, W, S]|value `0` for background, H, W can be `None` if ragged, can be one-hot format (`S` classes)|
|||`keypoints`||WIP|
|||`custom_annotations`||define by user|

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
