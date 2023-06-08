import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
from keras_cv.visualization.draw_bounding_boxes import draw_bounding_boxes


@st.cache_data
def download_images():
    """Images from torchvision repo.

    References:
        - `torchvision <https://github.com/pytorch/vision>`_
    """
    astronaut = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/astronaut.jpg"
    dog2 = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg"
    person1 = "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/person1.jpg"
    default_images = {
        "astronaut": {
            "image": requests.get(astronaut).content,
            "boxes": pd.DataFrame(
                [
                    {"x": 20, "y": 10, "w": 350, "h": 500},
                ]
            ),
        },
        "dog2": {
            "image": requests.get(dog2).content,
            "boxes": pd.DataFrame(
                [
                    {"x": 75, "y": 135, "w": 385, "h": 360},
                ]
            ),
        },
        "person1": {
            "image": requests.get(person1).content,
            "boxes": pd.DataFrame(
                [
                    {"x": 125, "y": 180, "w": 200, "h": 380},
                ]
            ),
        },
    }
    return default_images


def construct_inputs(image, boxes=None):
    image = np.expand_dims(image.copy(), axis=0)
    inputs = {"images": tf.convert_to_tensor(image, dtype=tf.float32)}
    if boxes is not None:
        inputs["bounding_boxes"] = {}
        boxes = tf.convert_to_tensor(
            np.expand_dims(boxes, axis=0), dtype=tf.float32
        )
        inputs["bounding_boxes"]["boxes"] = boxes
        inputs["bounding_boxes"]["classes"] = tf.zeros(shape=boxes.shape[:-1])

    return inputs


def draw_bounding_boxes_on_inputs(inputs):
    # must make a copy to avoid modifying the original inputs
    bounding_boxes = inputs["bounding_boxes"].copy()
    drawed_images = draw_bounding_boxes(
        inputs["images"],
        bounding_boxes,
        color=(0, 188, 212),
        bounding_box_format="xywh",
    )
    return drawed_images


def process_inputs(inputs, layer):
    outputs = layer(inputs)
    if "bounding_boxes" in outputs:
        drawed_images = draw_bounding_boxes(
            outputs["images"],
            outputs["bounding_boxes"],
            color=(0, 188, 212),
            bounding_box_format="xywh",
        )
    else:
        drawed_images = np.array(outputs["images"]).astype(np.uint8)
    return drawed_images
