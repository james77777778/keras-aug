import io
import typing

import tensorflow as tf
import numpy as np
import requests
import streamlit as st
from layers_config import LAYERS_CONFIG
from PIL import Image


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
        "astronaut": requests.get(astronaut).content,
        "dog2": requests.get(dog2).content,
        "person1": requests.get(person1).content,
    }
    return default_images


def set_control_args(control_args: typing.Dict, layer_args: typing.Dict):
    for key, value in control_args.items():
        min_value = value[0]
        max_value = value[1]
        value = layer_args[key]
        new_value = st.slider(key, min_value, max_value, value)
        layer_args[key] = new_value
    return layer_args


def transform_image(image, layer):
    processed_image = layer(image)
    processed_image: np.ndarray = processed_image.numpy()
    processed_image = np.round(processed_image).astype(np.uint8)
    return processed_image


def main():
    data_load_state = st.text("Loading data...")
    default_images = download_images()
    data_load_state.empty()
    with st.sidebar:
        # images
        image_option = st.selectbox(
            "Select the default image",
            list(default_images.keys()),
            label_visibility="collapsed",
        )
        with st.expander("Upload an image"):
            uploaded_image = st.file_uploader(
                "", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
            )
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
        else:
            image = Image.open(io.BytesIO(default_images[image_option]))
        image = np.array(image)

        # layers
        init_index = 1
        layer_option = st.selectbox(
            "Select the KerasAug's layer",
            list(LAYERS_CONFIG.keys()),
            init_index,
        )
        layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
        layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
        control_args = LAYERS_CONFIG[layer_option]["control_args"]
        layer_args = set_control_args(control_args, layer_args)
        layer = layer_cls(**layer_args)

    st.text('Press "R" to rerun')
    col1, _, col3 = st.columns([0.45, 0.1, 0.45], gap="large")
    with col1:
        st.text("Original")
        st.image(image, use_column_width=True)
    with col3:
        st.text("Processed")
        st.image(transform_image(image, layer), use_column_width=True)


if __name__ == "__main__":
    # disable GPU
    try:
        tf.config.set_visible_devices([], "GPU")
    except:
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
