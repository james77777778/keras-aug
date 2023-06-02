import io

import keras_cv
import requests
from PIL import Image

import keras_aug
import streamlit as st


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


def get_layer(layer_cls, layer_args):
    pass


def transform_image(image, layer):
    pass


def main():
    data_load_state = st.text("Loading data...")
    default_images = download_images()
    data_load_state.empty()
    with st.sidebar:
        st.title("Options")
        option = st.selectbox(
            "Select the default image",
            list(default_images.keys()),
        )
        uploaded_image = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
        else:
            image = Image.open(io.BytesIO(default_images[option]))

    col1, col2 = st.columns(2)
    with col1:
        st.text("Original")
        st.image(image, use_column_width=True)
    with col2:
        st.text("Processed")
        st.image(image, use_column_width=True)


if __name__ == "__main__":
    st.set_page_config(
        page_title="KerasAug Demo Site",
        initial_sidebar_state="expanded",
    )
    main()
