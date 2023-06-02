import keras_aug
import streamlit as st

LAYERS_CONFIG = {
    "AugMix": {
        "layer_cls": keras_aug.layers.AugMix,
        "layer_args": {
            "value_range": (0, 255),
            "severity": [0.01, 0.3],
            "num_chains": 3,
            "chain_depth": [1, 3],
            "alpha": 1.0,
        },
        "control_args": {
            "severity": [0.01, 1.0],
            "num_chains": [1, 5],
            "chain_depth": [1, 5],
            "alpha": [0.01, 2.0],
        },
        "is_compatible_with_bbox": False,
    },
    "RandAugment": {
        "layer_cls": keras_aug.layers.RandAugment,
        "layer_args": {
            "value_range": (0, 255),
            "augmentations_per_image": 2,
            "magnitude": 10.0,
            "magnitude_stddev": 0.0,
            "translation_multiplier": 150.0 / 331.0,
            "use_geometry": 1,
            "interpolation": "nearest",
            "fill_mode": "reflect",
            "fill_value": 0,
            "exclude_ops": None,
            "bounding_box_format": None,
        },
        "control_args": {
            "augmentations_per_image": [1, 3],
            "magnitude": [0.0, 30.0],
            "use_geometry": [0, 1],
        },
        "is_compatible_with_bbox": True,
    },
}
