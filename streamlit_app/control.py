import io
import typing

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


def control_images(default_images):
    image_option = st.selectbox("Default images", list(default_images.keys()))
    with st.expander("Upload an image"):
        uploaded_image = st.file_uploader(
            "dummy",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        pd_boxes = pd.DataFrame([{"x": 0, "y": 0, "w": 0, "h": 0}])  # default
        mask = np.zeros_like(image)
    else:
        image = Image.open(
            io.BytesIO(default_images[image_option]["image"])
        ).convert("RGB")
        pd_boxes = default_images[image_option]["boxes"]
        mask = default_images[image_option]["mask"]
    image = np.array(image)
    return image, pd_boxes, mask


def control_bounding_boxes(pd_boxes):
    show_bounding_boxes = st.checkbox("Show bounding boxes", value=False)
    if show_bounding_boxes:
        st.text("You can edit them")
        pd_boxes = st.data_editor(
            pd_boxes,
            num_rows="dynamic",
            column_config={
                "x": st.column_config.NumberColumn(default=0, format="%d"),
                "y": st.column_config.NumberColumn(default=0, format="%d"),
                "w": st.column_config.NumberColumn(default=0, format="%d"),
                "h": st.column_config.NumberColumn(default=0, format="%d"),
            },
            hide_index=True,
        )
    boxes = np.array(pd_boxes.values.tolist())
    return show_bounding_boxes, boxes


def control_segmentation_masks(mask):
    mask = np.array(mask)
    show_segmentation_mask = st.checkbox("Show segmentation mask", value=False)
    if show_segmentation_mask:
        with st.expander("Upload an segmentation mask"):
            uploaded_segmentation_mask = st.file_uploader(
                "Segmentation mask",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        if uploaded_segmentation_mask is not None:
            mask = np.array(uploaded_segmentation_mask)
    return show_segmentation_mask, mask


def set_control_args(control_args: typing.Dict, layer_args: typing.Dict):
    """Use `st.select_slider` or `st.slider` for `control_args` depending on
    default value.
    """
    with st.form(key="control"):
        new_values = {}
        for key, value in control_args.items():
            if isinstance(layer_args[key], str):
                options = value
                default_value = layer_args[key]
                new_value = st.select_slider(key, options, default_value)
            else:
                min_value = value[0]
                max_value = value[1]
                default_value = layer_args[key]
                new_value = st.slider(key, min_value, max_value, default_value)
            new_values[key] = new_value
        submit_button = st.form_submit_button(label="Apply")
        if submit_button:
            layer_args.update(new_values)
    return layer_args
