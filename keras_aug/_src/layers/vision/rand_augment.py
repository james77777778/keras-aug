import typing

import keras
import numpy as np
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_bbox_format
from keras_aug._src.utils.argument_validation import standardize_interpolation
from keras_aug._src.utils.argument_validation import standardize_padding_mode


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandAugment(VisionRandomLayer):
    """RandAugment data augmentation method.

    Note that due to implementation limitations, the randomness occurs in a
    batch manner.

    References:
    - [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719)

    Args:
        num_ops: Number of augmentation transformations to apply sequentially.
            Defaults to `2`.
        magnitude: Magnitude for all the transformations. Defaults to `9`.
        num_magnitude_bins: The number of different magnitude values. Defaults
            to `31`.
        geometric: Whether to include geometric augmentations. This
            can be set to `False` when the inputs containing bounding boxes.
            Defaults to `True`.
        interpolation: The interpolation mode. Available values are:
            `"nearest", "bilinear"`. Defaults to `"bilinear"`.
        padding_mode: The mode of the padding. Available values:
            `"constant", "edge", "reflect", "symmetric"`. Defaults to
            `"constant"`.
        padding_value: The padding value. It only affects when
            `padding_mode="constant"`. Defaults to `0`.
        bounding_box_format: The format of the bounding boxes. If specified,
            the available values are `"xyxy", "xywh", "center_xywh", "rel_xyxy",
            "rel_xywh", "rel_center_xywh"`. Defaults to `None`.
        data_format: A string specifying the data format of the input images.
            It can be either `"channels_last"` or `"channels_first"`.
            If not specified, the value will be interpreted by
            `keras.config.image_data_format`. Defaults to `None`.
    """  # noqa: E501

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        geometric: bool = True,
        interpolation: str = "bilinear",
        padding_mode: str = "constant",
        padding_value: float = 0,
        bounding_box_format: typing.Optional[str] = None,
        data_format=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Check
        if magnitude >= num_magnitude_bins or magnitude < 0:
            raise ValueError(
                "`magnitude` must be a positive value and lower that "
                "`num_magnitude_bins`. "
                f"Received: magnitude={magnitude}, "
                f"num_magnitude_bins={num_magnitude_bins}"
            )

        self.num_ops = int(num_ops)
        self.magnitude = int(magnitude)
        self.num_magnitude_bins = int(num_magnitude_bins)
        self.geometric = bool(geometric)
        self.interpolation = standardize_interpolation(interpolation)
        self.padding_mode = standardize_padding_mode(padding_mode)
        self.padding_value = padding_value
        self.bounding_box_format = standardize_bbox_format(bounding_box_format)
        self.data_format = data_format or keras.config.image_data_format()

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1
        num_bins = self.num_magnitude_bins
        self.augmentation_space = {
            "Identity": None,
            "Brightness": np.linspace(0.0, 0.9, num_bins, dtype="float32"),
            "Color": np.linspace(0.0, 0.9, num_bins, dtype="float32"),
            "Contrast": np.linspace(0.0, 0.9, num_bins, dtype="float32"),
            "Sharpness": np.linspace(0.0, 0.9, num_bins, dtype="float32"),
            "Posterize": (
                (8 - (np.arange(num_bins) / ((num_bins - 1) / 4)))
                .round()
                .astype("int32")
            ),
            "Solarize": np.linspace(1.0, 0.0, num_bins, dtype="float32"),
            "AutoContrast": None,
            "Equalize": None,
        }
        if self.geometric:
            self.augmentation_space.update(
                {
                    "ShearX": np.degrees(
                        np.arctan(np.linspace(0.0, 0.3, num_bins))
                    ).astype("float32"),
                    "ShearY": np.degrees(
                        np.arctan(np.linspace(0.0, 0.3, num_bins))
                    ).astype("float32"),
                    "TranslateX": np.linspace(
                        0.0, 150.0 / 331.0, num_bins, dtype="float32"
                    ),
                    "TranslateY": np.linspace(
                        0.0, 150.0 / 331.0, num_bins, dtype="float32"
                    ),
                    "Rotate": np.linspace(0.0, 30.0, num_bins, dtype="float32"),
                }
            )
        p = [1.0] * len(self.augmentation_space)
        total = sum(p)
        self.p = [prob / total for prob in p]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        magnitude = ops.numpy.full(
            [self.num_ops, batch_size], self.magnitude, dtype="int32"
        )
        fn_idx_p = ops.convert_to_tensor([self.p])
        fn_idx = ops.random.categorical(
            ops.numpy.log(fn_idx_p), self.num_ops, seed=random_generator
        )
        fn_idx = fn_idx[0]
        signed_p = ops.random.uniform([batch_size]) > 0.5
        signed = ops.cast(ops.numpy.where(signed_p, 1.0, -1.0), dtype="float32")
        return dict(
            magnitude=magnitude,  # shape: (self.num_ops, batch_size)
            fn_idx=fn_idx,  # shape: (self.num_ops,)
            signed=signed,  # shape: (batch_size,)
        )

    def _apply_images_transform(self, images, magnitude, idx, signed):
        ops = self.backend

        dtype = backend.standardize_dtype(images.dtype)
        batch_size = ops.shape(images)[0]

        # Build branches for ops.switch
        aug_space = self.augmentation_space
        max_value = self.image_backend._max_value_of_dtype(dtype)
        transforms = []
        for key in sorted(self.augmentation_space.keys()):
            if key == "Identity":
                transforms.append(lambda x: x)
            elif key == "Brightness":
                factor = ops.numpy.add(
                    1.0,
                    ops.numpy.multiply(
                        signed,
                        ops.numpy.take(aug_space["Brightness"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.image_backend.adjust_brightness(x, factor)
                )
            elif key == "Color":
                factor = ops.numpy.add(
                    1.0,
                    signed * ops.numpy.take(aug_space["Color"], magnitude),
                )
                transforms.append(
                    lambda x: self.image_backend.adjust_saturation(x, factor)
                )
            elif key == "Contrast":
                factor = ops.numpy.add(
                    1.0,
                    signed * ops.numpy.take(aug_space["Contrast"], magnitude),
                )
                transforms.append(
                    lambda x: self.image_backend.adjust_contrast(
                        x, factor, data_format=self.data_format
                    )
                )
            elif key == "Sharpness":
                factor = ops.numpy.add(
                    1.0,
                    signed * ops.numpy.take(aug_space["Sharpness"], magnitude),
                )
                transforms.append(
                    lambda x: self.image_backend.sharpen(
                        x, factor, data_format=self.data_format
                    )
                )
            elif key == "Posterize":
                bits = ops.numpy.take(aug_space["Posterize"], magnitude)
                transforms.append(
                    lambda x: self.image_backend.posterize(x, bits)
                )
            elif key == "Solarize":
                factor = ops.numpy.multiply(
                    max_value, ops.numpy.take(aug_space["Solarize"], magnitude)
                )
                transforms.append(
                    lambda x: self.image_backend.solarize(x, factor)
                )
            elif key == "AutoContrast":
                transforms.append(
                    lambda x: self.image_backend.auto_contrast(
                        x, data_format=self.data_format
                    )
                )
            elif key == "Equalize":
                transforms.append(
                    lambda x: self.image_backend.equalize(
                        x, data_format=self.data_format
                    )
                )
            elif key == "ShearX":
                shear_x = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["ShearX"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.image_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        shear_x,
                        ops.numpy.zeros([batch_size]),
                        interpolation=self.interpolation,
                        padding_mode=self.padding_mode,
                        padding_value=self.padding_value,
                        data_format=self.data_format,
                    )
                )
            elif key == "ShearY":
                shear_y = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["ShearY"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.image_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        shear_y,
                        interpolation=self.interpolation,
                        padding_mode=self.padding_mode,
                        padding_value=self.padding_value,
                        data_format=self.data_format,
                    )
                )
            elif key == "TranlateX":
                translate_x = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["TranlateX"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.image_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        translate_x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        interpolation=self.interpolation,
                        padding_mode=self.padding_mode,
                        padding_value=self.padding_value,
                        data_format=self.data_format,
                    )
                )
            elif key == "TranlateY":
                translate_y = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["TranlateY"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.image_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        translate_y,
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        interpolation=self.interpolation,
                        padding_mode=self.padding_mode,
                        padding_value=self.padding_value,
                        data_format=self.data_format,
                    )
                )
            elif key == "Rotate":
                angle = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["Rotate"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.image_backend.affine(
                        x,
                        angle,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        interpolation=self.interpolation,
                        padding_mode=self.padding_mode,
                        padding_value=self.padding_value,
                        data_format=self.data_format,
                    )
                )
        images = ops.core.switch(idx, transforms, images)
        return images

    def augment_images(self, images, transformations, **kwargs):
        magnitude = transformations["magnitude"]
        fn_idx = transformations["fn_idx"]
        signed = transformations["signed"]
        for i in range(self.num_ops):
            idx = fn_idx[i]
            m = magnitude[i]
            images = self._apply_images_transform(images, m, idx, signed)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def _apply_bounding_boxes_transform(
        self, bounding_boxes, height, width, magnitude, idx, signed
    ):
        ops = self.backend

        batch_size = ops.shape(bounding_boxes["boxes"])[0]

        # Build branches for ops.switch
        aug_space = self.augmentation_space
        transforms = []
        for key in sorted(self.augmentation_space.keys()):
            if key == "Identity":
                transforms.append(lambda x: x)
            elif key == "Brightness":
                transforms.append(lambda x: x)
            elif key == "Color":
                transforms.append(lambda x: x)
            elif key == "Contrast":
                transforms.append(lambda x: x)
            elif key == "Sharpness":
                transforms.append(lambda x: x)
            elif key == "Posterize":
                transforms.append(lambda x: x)
            elif key == "Solarize":
                transforms.append(lambda x: x)
            elif key == "AutoContrast":
                transforms.append(lambda x: x)
            elif key == "Equalize":
                transforms.append(lambda x: x)
            elif key == "ShearX":
                shear_x = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["ShearX"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.bbox_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        shear_x,
                        ops.numpy.zeros([batch_size]),
                        height=height,
                        width=width,
                    )
                )
            elif key == "ShearY":
                shear_y = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["ShearY"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.bbox_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        shear_y,
                        height=height,
                        width=width,
                    )
                )
            elif key == "TranlateX":
                translate_x = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["TranlateX"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.bbox_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        translate_x,
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        height=height,
                        width=width,
                    )
                )
            elif key == "TranlateY":
                translate_y = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["TranlateY"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.bbox_backend.affine(
                        x,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        translate_y,
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        height=height,
                        width=width,
                    )
                )
            elif key == "Rotate":
                angle = ops.numpy.multiply(
                    signed,
                    ops.numpy.full(
                        [batch_size],
                        ops.numpy.take(aug_space["Rotate"], magnitude),
                    ),
                )
                transforms.append(
                    lambda x: self.bbox_backend.affine(
                        x,
                        angle,
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.ones([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        ops.numpy.zeros([batch_size]),
                        height=height,
                        width=width,
                    )
                )
        boxes = ops.core.switch(idx, transforms, bounding_boxes["boxes"])
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
        return bounding_boxes

    def augment_bounding_boxes(
        self,
        bounding_boxes,
        transformations,
        images=None,
        raw_images=None,
        **kwargs,
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                f"{self.__class__.__name__} was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
            )
        ops = self.backend

        magnitude = transformations["magnitude"]
        fn_idx = transformations["fn_idx"]
        signed = transformations["signed"]
        images_shape = ops.shape(raw_images)
        height = images_shape[self.h_axis]
        width = images_shape[self.w_axis]
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=height,
            width=width,
            dtype=self.bounding_box_dtype,
        )
        for i in range(self.num_ops):
            idx = fn_idx[i]
            m = magnitude[i]
            bounding_boxes = self._apply_bounding_boxes_transform(
                bounding_boxes, height, width, m, idx, signed
            )
        bounding_boxes = self.bbox_backend.clip_to_images(
            bounding_boxes,
            height=height,
            width=width,
            format="xyxy",
        )
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=height,
            width=width,
            dtype=self.bounding_box_dtype,
        )
        return bounding_boxes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_ops": self.num_ops,
                "magnitude": self.magnitude,
                "num_magnitude_bins": self.num_magnitude_bins,
                "geometric": self.geometric,
                "interpolation": self.interpolation,
                "padding_mode": self.padding_mode,
                "padding_value": self.padding_value,
            }
        )
        return config
