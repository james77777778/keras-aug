import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import tensorflow as tf
import keras_aug
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
    astronaut_mask = "./streamlit_app/assets/astronaut.png"
    dog2_mask = "./streamlit_app/assets/dog2.png"
    person1_mask = "./streamlit_app/assets/person1.png"
    default_images = {
        "astronaut": {
            "image": requests.get(astronaut).content,
            "boxes": pd.DataFrame(
                [
                    {"x": 20, "y": 10, "w": 350, "h": 500},
                ]
            ),
            "mask": cv2.imread(astronaut_mask),
        },
        "dog2": {
            "image": requests.get(dog2).content,
            "boxes": pd.DataFrame(
                [
                    {"x": 75, "y": 135, "w": 385, "h": 360},
                ]
            ),
            "mask": cv2.imread(dog2_mask),
        },
        "person1": {
            "image": requests.get(person1).content,
            "boxes": pd.DataFrame(
                [
                    {"x": 125, "y": 180, "w": 200, "h": 380},
                ]
            ),
            "mask": cv2.imread(person1_mask),
        },
    }
    return default_images


def construct_inputs(image, boxes=None, mask=None):
    image = np.expand_dims(image.copy(), axis=0)
    inputs = {"images": tf.convert_to_tensor(image, dtype=tf.float32)}
    if boxes is not None:
        inputs["bounding_boxes"] = {}
        boxes = tf.convert_to_tensor(
            np.expand_dims(boxes, axis=0), dtype=tf.float32
        )
        inputs["bounding_boxes"]["boxes"] = boxes
        inputs["bounding_boxes"]["classes"] = tf.zeros(shape=boxes.shape[:-1])
    if mask is not None:
        inputs["segmentation_masks"] = tf.convert_to_tensor(
            np.expand_dims(mask, axis=0), dtype=tf.float32
        )
    return inputs


def draw_bounding_boxes_app(inputs):
    # must make a copy to avoid modifying the original inputs
    bounding_boxes = inputs["bounding_boxes"].copy()
    drawed_images = draw_bounding_boxes(
        inputs["images"],
        bounding_boxes,
        color=(0, 188, 212),
        bounding_box_format="xywh",
    )
    return drawed_images


def draw_segmentation_masks_app(inputs):
    # must make a copy to avoid modifying the original inputs
    drawed_images = keras_aug.datapoints.image.blend(
        inputs["images"],
        inputs["segmentation_masks"],
        0.25,
        (0, 255),
    )
    return drawed_images
