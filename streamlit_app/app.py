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


def process_image(image, layer):
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
        # set to 1 to select RandAugment
        layer_option = st.selectbox(
            "", list(LAYERS_CONFIG.keys()), 1, label_visibility="collapsed"
        )
        layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
        layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
        control_args = LAYERS_CONFIG[layer_option]["control_args"]
        layer_args = set_control_args(control_args, layer_args)
        layer = layer_cls(**layer_args)

    st.text('Press "R" to generate new random image')
    col1, _, col3 = st.columns([0.45, 0.1, 0.45], gap="large")
    with col1:
        st.text(f"Original {image.shape}")
        st.image(image, use_column_width=True)
    with col3:
        processed_image = process_image(image, layer)
        st.text(f"Processed {processed_image.shape}")
        st.image(processed_image, use_column_width=True)

    # show help
    with st.expander(f"Click to display help for {layer_cls.__name__}"):
        st.help(layer)


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
