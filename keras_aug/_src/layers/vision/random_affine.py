import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_bbox_format
from keras_aug._src.utils.argument_validation import standardize_data_format
from keras_aug._src.utils.argument_validation import standardize_interpolation
from keras_aug._src.utils.argument_validation import standardize_padding_mode
from keras_aug._src.utils.argument_validation import standardize_parameter


@keras_aug_export(parent_path=["keras_aug.layers.vision"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomAffine(VisionRandomLayer):
    """Random affine transformation the inputs keeping center invariant.

    Note that `degree` and `shear` are in angle units, while `translate`,
    `scale` and `center` are in percentage units.

    Args:
        degree: Range of degrees to select from. If `degree` is a number instead
            of sequence like `(min, max)`, the range of degrees will be
            `(-degree, +degree)`. Set to `None` to deactivate rotations.
            Defaults to `None`.
        translate: Sequence of maximum absolute fraction for horizontal and
            vertical translations. For example `(0.1, 0.2)`, then horizontal
            shift is randomly sampled in the range of
            `(-width * 0.1, width * 0.1)` and vertical shift is randomly
            sampled in the range of `(-height * 0.2, height * 0.2)`. Set to
            `None` to deactivate translations. Defaults to `None`.
        scale: Range of scales to select from. Set to `None` to deactivate
            scalings. Defaults to `None`.
        shear: Range of degrees to select from. If `shear` is a number a shear
            parallel to the x-axis in the range of `(-shear, shear)` will be
            applied. Else if `shear` is a sequence of 2, a shear parallel to the
            x-axis in the range of `(shear[0], shear[1])` will be applied. Else
            if `shear` is a sequence of 4, a shear parallel to the x-axis in the
            range of `(shear[0], shear[1])`and y-axis shear in the range of
            `(shear[2], shear[3])` will be applied. Set to `None` to deactivate
            shear. Defaults to `None`.
        center: Optional center of rotation in the format of `(rel_x, rel_y)`.
            Default is the center of the images.
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
    """

    def __init__(
        self,
        degree: typing.Union[None, float, typing.Sequence[float]] = None,
        translate: typing.Union[None, float, typing.Sequence[float]] = None,
        scale: typing.Union[None, float, typing.Sequence[float]] = None,
        shear: typing.Union[None, float, typing.Sequence[float]] = None,
        center: typing.Optional[typing.Sequence[float]] = None,
        interpolation: str = "bilinear",
        padding_mode: str = "constant",
        padding_value: float = 0,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        translate, shear, center, padding_mode = self._check_arguments(
            translate, shear, center, padding_mode
        )
        self.degree = standardize_parameter(degree)
        self.translate = translate
        self.scale = standardize_parameter(scale)
        self.shear = shear
        self.center = center
        self.interpolation = standardize_interpolation(interpolation)
        self.padding_mode = standardize_padding_mode(padding_mode)
        self.padding_value = padding_value
        self.bounding_box_format = standardize_bbox_format(bounding_box_format)
        self.data_format = standardize_data_format(data_format)

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def _check_arguments(self, translate, shear, center, padding_mode):
        if translate is not None:
            if isinstance(translate, float):
                translate = (translate, translate)
            translate = tuple(translate)
            if len(translate) != 2:
                raise ValueError(
                    "If `translate` is specified, it must be a sequence of 2 "
                    f"values. Received: translate={translate}"
                )
            if (
                translate[0] < 0
                or translate[0] > 1
                or translate[1] < 0
                or translate[1] > 1
            ):
                raise ValueError(
                    "If `translate` is specified, the value range must be in "
                    f"`[0, 1]`. Received: translate={translate}"
                )
        if shear is not None:
            shear = tuple(shear)
            if len(shear) not in (2, 4):
                raise ValueError(
                    "If `shear` is specified, it must be a sequence of 2 or 4 "
                    f"values. Received: shear={shear}"
                )
        if center is not None:
            center = tuple(float(c) for c in center)
            if len(center) != 2:
                raise ValueError(
                    "If `center` is specified, it must be a sequence of 2 "
                    f"values. Received: center={center}"
                )

        available_padding_mode = ("constant", "reflect", "symmetric")
        if padding_mode not in available_padding_mode:
            raise ValueError(
                "Invalid `padding_mode`. Available values are: "
                f"{list(available_padding_mode)}. "
                f"Received: padding_mode={padding_mode}"
            )
        return translate, shear, center, padding_mode

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator

        def generate_params(low, high):
            return ops.random.uniform(
                [batch_size], low, high, seed=random_generator
            )

        if self.scale is not None:
            scale = generate_params(self.scale[0], self.scale[1])
        else:
            scale = ops.numpy.ones([batch_size])
        if self.degree is not None:
            angle = generate_params(self.degree[0], self.degree[1])
        else:
            angle = ops.numpy.zeros([batch_size])
        if self.shear is not None:
            shear_x = generate_params(self.shear[0], self.shear[1])
            if len(self.shear) == 4:
                shear_y = generate_params(self.shear[2], self.shear[3])
            else:
                shear_y = ops.numpy.zeros([batch_size])
        else:
            shear_x = ops.numpy.zeros([batch_size])
            shear_y = ops.numpy.zeros([batch_size])
        if self.translate is not None:
            translate_x = generate_params(-self.translate[0], self.translate[0])
            translate_y = generate_params(-self.translate[1], self.translate[1])
        else:
            translate_x = ops.numpy.zeros([batch_size])
            translate_y = ops.numpy.zeros([batch_size])
        return dict(
            angle=angle,
            translate_x=translate_x,
            translate_y=translate_y,
            scale=scale,
            shear_x=shear_x,
            shear_y=shear_y,
        )

    def augment_images(self, images, transformations, **kwargs):
        if self.center is None:
            center_x, center_y = 0.5, 0.5
        else:
            center_x, center_y = self.center
        images = self.image_backend.affine(
            images,
            transformations["angle"],
            transformations["translate_x"],
            transformations["translate_y"],
            transformations["scale"],
            transformations["shear_x"],
            transformations["shear_y"],
            center_x,
            center_y,
            self.interpolation,
            self.padding_mode,
            self.padding_value,
            self.data_format,
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

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
        boxes = self.bbox_backend.affine(
            bounding_boxes["boxes"],
            transformations["angle"],
            transformations["translate_x"],
            transformations["translate_y"],
            transformations["scale"],
            transformations["shear_x"],
            transformations["shear_y"],
            height,
            width,
        )
        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = boxes
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

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        original_dtype = segmentation_masks.dtype
        if self.center is None:
            center_x, center_y = 0.5, 0.5
        else:
            center_x, center_y = self.center

        segmentation_masks = ops.cast(
            segmentation_masks,
            backend.result_type(segmentation_masks.dtype, float),
        )
        segmentation_masks = self.image_backend.affine(
            segmentation_masks,
            transformations["angle"],
            transformations["translate_x"],
            transformations["translate_y"],
            transformations["scale"],
            transformations["shear_x"],
            transformations["shear_y"],
            center_x,
            center_y,
            "nearest",
            "constant",
            -1,
            self.data_format,
        )
        return ops.cast(segmentation_masks, original_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "degree": self.degree,
                "translate": self.translate,
                "scale": self.scale,
                "shear": self.shear,
                "center": self.center,
                "interpolation": self.interpolation,
                "padding_mode": self.padding_mode,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config
