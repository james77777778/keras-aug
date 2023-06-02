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
    "TrivialAugmentWide": {
        "layer_cls": keras_aug.layers.TrivialAugmentWide,
        "layer_args": {
            "value_range": (0, 255),
            "use_geometry": 1,
        },
        "control_args": {
            "use_geometry": [0, 1],
        },
        "is_compatible_with_bbox": True,
    },
    "RandomAffine": {
        "layer_cls": keras_aug.layers.RandomAffine,
        "layer_args": {
            "rotation_factor": (-10.0, 10.0),
            "translation_height_factor": (-0.1, 0.1),
            "translation_width_factor": (-0.1, 0.1),
            "zoom_height_factor": (0.5, 1.5),
            "zoom_width_factor": (0.5, 1.5),
            "shear_height_factor": (-0.1, 0.1),
            "shear_width_factor": (-0.1, 0.1),
            "same_zoom_factor": 1,
        },
        "control_args": {
            "rotation_factor": [-90.0, 90.0],
            "translation_height_factor": [-0.5, 0.5],
            "translation_width_factor": [-0.5, 0.5],
            "zoom_height_factor": (0.01, 2.0),
            "zoom_width_factor": (0.01, 2.0),
            "shear_height_factor": [-0.5, 0.5],
            "shear_width_factor": [-0.5, 0.5],
            "same_zoom_factor": [0, 1],
        },
    },
    "RandomCrop": {
        "layer_cls": keras_aug.layers.RandomCrop,
        "layer_args": {
            "height": 150,
            "width": 150,
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
        },
    },
    "RandomCropAndResize": {
        "layer_cls": keras_aug.layers.RandomCropAndResize,
        "layer_args": {
            "height": 200,
            "width": 200,
            "crop_area_factor": (0.8, 1.0),
            "aspect_ratio_factor": (3 / 4, 4 / 3),
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
            "crop_area_factor": (0.1, 1.0),
            "aspect_ratio_factor": (0.5, 2.0),
        },
    },
    "RandomFlip": {
        "layer_cls": keras_aug.layers.RandomFlip,
        "layer_args": {
            "mode": "horizontal",
        },
        "control_args": {
            "mode": ["horizontal", "vertical", "horizontal_and_vertical"],
        },
    },
    # skip RandomResize
    "RandomRotate": {
        "layer_cls": keras_aug.layers.RandomRotate,
        "layer_args": {
            "factor": (-10.0, 10.0),
        },
        "control_args": {
            "factor": (-90.0, 90.0),
        },
    },
    "RandomZoomAndCrop": {
        "layer_cls": keras_aug.layers.RandomZoomAndCrop,
        "layer_args": {
            "height": 200,
            "width": 200,
            "scale_factor": (0.8, 1.25),
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
            "scale_factor": (0.1, 10.0),
        },
    },
    "ChannelShuffle": {
        "layer_cls": keras_aug.layers.ChannelShuffle,
        "layer_args": {
            "groups": 3,
        },
        "control_args": {
            "groups": [1, 3],
        },
    },
    "RandomBlur": {
        "layer_cls": keras_aug.layers.RandomBlur,
        "layer_args": {
            "factor": (3, 7),
        },
        "control_args": {
            "factor": [1, 99],
        },
    },
    "RandomChannelShift": {
        "layer_cls": keras_aug.layers.RandomChannelShift,
        "layer_args": {
            "value_range": (0, 255),
            "factor": (-0.1, 0.1),
            "channels": 3,
        },
        "control_args": {
            "factor": [-1.0, 1.0],
        },
    },
    "RandomCLAHE": {
        "layer_cls": keras_aug.layers.RandomCLAHE,
        "layer_args": {
            "value_range": (0, 255),
            "factor": (4, 4),
            "tile_grid_size": (8, 8),
        },
        "control_args": {
            "factor": (1, 10),
            "tile_grid_size": (2, 16),
        },
    },
    "RandomColorJitter": {
        "layer_cls": keras_aug.layers.RandomColorJitter,
        "layer_args": {
            "value_range": (0, 255),
            "brightness_factor": (0.5, 1.5),
            "contrast_factor": (0.5, 1.5),
            "saturation_factor": (0.5, 1.5),
            "hue_factor": (-0.1, 0.1),
        },
        "control_args": {
            "brightness_factor": (0.1, 2.0),
            "contrast_factor": (0.1, 2.0),
            "saturation_factor": (0.1, 2.0),
            "hue_factor": (-0.5, 0.5),
        },
    },
    "RandomGamma": {
        "layer_cls": keras_aug.layers.RandomGamma,
        "layer_args": {
            "value_range": (0, 255),
            "factor": (0.5, 1.5),
        },
        "control_args": {
            "factor": (0.1, 2.0),
        },
    },
    "RandomGaussianBlur": {
        "layer_cls": keras_aug.layers.RandomGaussianBlur,
        "layer_args": {
            "kernel_size": 7,
            "factor": (1.0, 1.0),
        },
        "control_args": {
            "kernel_size": (1, 99),
            "factor": (0.0, 2.0),
        },
    },
    "RandomHSV": {
        "layer_cls": keras_aug.layers.RandomHSV,
        "layer_args": {
            "value_range": (0, 255),
            "hue_factor": (-0.1, 0.1),
            "saturation_factor": (0.5, 1.5),
            "value_factor": (0.5, 1.5),
        },
        "control_args": {
            "hue_factor": (-0.5, 0.5),
            "saturation_factor": (0.1, 2.0),
            "value_factor": (0.1, 2.0),
        },
    },
    "RandomJpegQuality": {
        "layer_cls": keras_aug.layers.RandomJpegQuality,
        "layer_args": {
            "value_range": (0, 255),
            "factor": (50, 100),
        },
        "control_args": {
            "factor": (1, 100),
        },
    },
    "RandomPosterize": {
        "layer_cls": keras_aug.layers.RandomPosterize,
        "layer_args": {
            "value_range": (0, 255),
            "factor": (5, 8),
        },
        "control_args": {
            "factor": (1, 8),
        },
    },
    "RandomSharpness": {
        "layer_cls": keras_aug.layers.RandomSharpness,
        "layer_args": {
            "value_range": (0, 255),
            "factor": (1.5, 1.5),
        },
        "control_args": {
            "factor": (0.1, 2.0),
        },
    },
    "RandomSolarize": {
        "layer_cls": keras_aug.layers.RandomSolarize,
        "layer_args": {
            "value_range": (0, 255),
            "threshold_factor": (100, 100),
            "addition_factor": (0, 0),
        },
        "control_args": {
            "threshold_factor": (0, 255),
            "addition_factor": (0, 255),
        },
    },
    # skip CutMix
    # skip MixUp
    # skip Mosaic
    "RandomChannelDropout": {
        "layer_cls": keras_aug.layers.RandomChannelDropout,
        "layer_args": {
            "factor": (0, 2),
        },
        "control_args": {},
    },
    "RandomCutout": {
        "layer_cls": keras_aug.layers.RandomCutout,
        "layer_args": {
            "height_factor": (0.5, 0.5),
            "width_factor": (0.5, 0.5),
        },
        "control_args": {
            "height_factor": (0.1, 1.0),
            "width_factor": (0.1, 1.0),
        },
    },
    "RandomErase": {
        "layer_cls": keras_aug.layers.RandomErase,
        "layer_args": {
            "area_factor": (0.02, 0.4),
            "aspect_ratio_factor": (0.3, 1.0 / 0.3),
        },
        "control_args": {
            "area_factor": (0.01, 1.0),
            "aspect_ratio_factor": (0.1, 10.0),
        },
    },
    "RandomGridMask": {
        "layer_cls": keras_aug.layers.RandomGridMask,
        "layer_args": {
            "size_factor": (96 / 224, 224 / 224),
            "ratio_factor": (0.6, 0.6),
            "rotation_factor": (0.0, 0.0),
        },
        "control_args": {
            "size_factor": (0.1, 1.0),
            "ratio_factor": (0.1, 1.0),
            "rotation_factor": (-90.0, 90.0),
        },
    },
    # skip RandomApply
    # skip RandomChoice
    # skip RepeatedAugment
    # skip VectorizedBaseRandomLayer
    "CenterCrop": {
        "layer_cls": keras_aug.layers.CenterCrop,
        "layer_args": {
            "height": 200,
            "width": 200,
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
        },
    },
    # skip PadIfNeeded
    "Resize": {
        "layer_cls": keras_aug.layers.Resize,
        "layer_args": {
            "height": 200,
            "width": 200,
            "crop_to_aspect_ratio": 0,
            "pad_to_aspect_ratio": 1,
        },
        "control_args": {
            "height": [10, 512],
            "width": [10, 512],
            "pad_to_aspect_ratio": [0, 1],
        },
    },
    "AutoContrast": {
        "layer_cls": keras_aug.layers.AutoContrast,
        "layer_args": {
            "value_range": (0, 255),
        },
        "control_args": {},
    },
    "Equalize": {
        "layer_cls": keras_aug.layers.Equalize,
        "layer_args": {
            "value_range": (0, 255),
        },
        "control_args": {},
    },
    "Grayscale": {
        "layer_cls": keras_aug.layers.Grayscale,
        "layer_args": {},
        "control_args": {},
    },
    "Invert": {
        "layer_cls": keras_aug.layers.Invert,
        "layer_args": {
            "value_range": (0, 255),
        },
        "control_args": {},
    },
    "Normalize": {
        "layer_cls": keras_aug.layers.Normalize,
        "layer_args": {
            "value_range": (0, 255),
        },
        "control_args": {},
    },
    # skip Rescale
    # skip SanitizeBoundingBox
}
