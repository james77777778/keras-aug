import numpy as np
import streamlit as st
import tensorflow as tf
from layers_config import LAYERS_CONFIG
from control import control_bounding_boxes
from control import control_images
from control import control_segmentation_masks
from control import set_control_args
from utils import construct_inputs
from utils import download_images
from utils import draw_bounding_boxes_app
from utils import draw_segmentation_masks_app


def main():
    data_load_state = st.text("Loading data...")
    default_images = download_images()
    data_load_state.empty()

    # control
    with st.sidebar:
        image, pd_boxes, mask = control_images(default_images)
        show_bounding_boxes, boxes = control_bounding_boxes(pd_boxes)
        show_segmentation_mask, mask = control_segmentation_masks(mask)

        # layers
        # set to 3 to select RandomAffine
        layer_option = st.selectbox("Layers", list(LAYERS_CONFIG.keys()), 3)
        layer_cls = LAYERS_CONFIG[layer_option]["layer_cls"]
        layer_args = LAYERS_CONFIG[layer_option]["layer_args"]
        control_args = LAYERS_CONFIG[layer_option]["control_args"]
        is_compatible_with_bbox = LAYERS_CONFIG[layer_option][
            "is_compatible_with_bbox"
        ]
        is_compatible_with_masks = LAYERS_CONFIG[layer_option][
            "is_compatible_with_masks"
        ]
        layer_args = set_control_args(control_args, layer_args)
        layer = layer_cls(**layer_args)

    # process inputs
    inputs = construct_inputs(image, boxes=boxes.copy(), mask=mask.copy())
    if is_compatible_with_bbox and show_bounding_boxes:
        inputs["images"] = draw_bounding_boxes_app(inputs.copy())
    if is_compatible_with_masks and show_segmentation_mask:
        inputs["images"] = draw_segmentation_masks_app(inputs.copy())
    if not is_compatible_with_bbox:
        inputs.pop("bounding_boxes")
    if not is_compatible_with_masks:
        inputs.pop("segmentation_masks")
    outputs = layer(inputs.copy())
    draw_inputs = np.array(inputs["images"]).astype(np.uint8)
    drawed_outputs = np.array(outputs["images"]).astype(np.uint8)

    # display results
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
        st.text(f"Original {draw_inputs[0].shape}")
        st.image(draw_inputs[0], use_column_width=True)
    with col3:
        st.text(f"Processed {drawed_outputs[0].shape}")
        st.image(drawed_outputs[0], use_column_width=True)

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
