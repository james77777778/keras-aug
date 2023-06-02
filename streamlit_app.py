import pathlib
import random

import streamlit as st
from PIL import Image
from tensorflow import keras


@st.cache_data
def download_images():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = keras.utils.get_file(
        "flower_photos", origin=dataset_url, untar=True
    )
    data_dir = pathlib.Path(data_dir)
    return data_dir


def main():
    st.title("KerasAug Demo Site")

    data_load_state = st.text("Loading data...")
    data_dir = download_images()
    image_list = list(data_dir.glob("*/*.jpg"))
    random_image_path = random.choice(image_list)
    image = Image.open(random_image_path)
    data_load_state.text("Loading data...done!")

    st.sidebar.title("Layer")

    st.image(image, use_column_width=True)


if __name__ == "__main__":
    main()
