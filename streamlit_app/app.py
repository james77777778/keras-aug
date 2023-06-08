import io
import typing

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from layers_config import LAYERS_CONFIG
from PIL import Image
from utils import construct_inputs
from utils import download_images
from utils import draw_bounding_boxes_on_inputs
from utils import process_inputs


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


def main():
    data_load_state = st.text("Loading data...")
    default_images = download_images()
    data_load_state.empty()
    with st.sidebar:
        # images
        image_option = st.selectbox(
            "Default images", list(default_images.keys())
        )
        with st.expander("Upload an image"):
            uploaded_image = st.file_uploader(
                "dummy",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
            )
        if uploaded_image is not None:
            image = Image.open(uploaded_image).convert("RGB")
            boxes = pd.DataFrame([{"x": 0, "y": 0, "w": 0, "h": 0}])  # default
        else:
            image = Image.open(
                io.BytesIO(default_images[image_option]["image"])
            ).convert("RGB")
            boxes = default_images[image_option]["boxes"]
        image = np.array(image)

        # bounding boxes
        show_bounding_boxes = st.checkbox("Show bounding boxes", value=False)
        if show_bounding_boxes:
            st.text("You can edit them")
            boxes = st.data_editor(
                boxes,
                num_rows="dynamic",
                column_config={
                    "x": st.column_config.NumberColumn(default=0, format="%d"),
                    "y": st.column_config.NumberColumn(default=0, format="%d"),
                    "w": st.column_config.NumberColumn(default=0, format="%d"),
                    "h": st.column_config.NumberColumn(default=0, format="%d"),
                },
                hide_index=True,
            )
        np_boxes = np.array(boxes.values.tolist())

        # layers
        # set to 1 to select RandAugment
        layer_option = st.selectbox("Layers", list(LAYERS_CONFIG.keys()), 1)
        layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
        layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
        control_args = LAYERS_CONFIG[layer_option]["control_args"]
        is_compatible_with_bbox = LAYERS_CONFIG[layer_option][
            "is_compatible_with_bbox"
        ]
        layer_args = set_control_args(control_args, layer_args)
        layer = layer_cls(**layer_args)

    # process inputs
    if is_compatible_with_bbox and show_bounding_boxes:
        inputs = construct_inputs(image, np_boxes.copy())
        images = draw_bounding_boxes_on_inputs(inputs.copy())
    else:
        inputs = construct_inputs(image)
        images = np.array(inputs["images"]).astype(np.uint8)
    outputs = process_inputs(inputs, layer)

    st.text(
        'Press "R" to generate new random image. '
        'Press "Apply" to apply new arguments.'
    )
    if show_bounding_boxes and not is_compatible_with_bbox:
        st.text(
            f"⚠️ {layer_cls.__name__} is not compatible with bounding boxes ⚠️"
        )
    col1, _, col3 = st.columns([0.45, 0.1, 0.45], gap="large")
    with col1:
        st.text(f"Original {images[0].shape}")
        st.image(images[0], use_column_width=True)
    with col3:
        st.text(f"Processed {outputs[0].shape}")
        st.image(outputs[0], use_column_width=True)

    # show help
    with st.expander(f"💡 Click to display help for {layer_cls.__name__}"):
        st.help(layer)


if __name__ == "__main__":
    # disable GPU
    try:
        tf.config.set_visible_devices([], "GPU")
    except (ValueError, RuntimeError):
        pass
    st.set_page_config(
        page_title="KerasAug Demo Site",
        initial_sidebar_state="expanded",
        menu_items={
            "Report a Bug": "https://github.com/james77777778/keras-aug/issues",
            "About": "https://github.com/james77777778/keras-aug",
        },
    )
    main()
