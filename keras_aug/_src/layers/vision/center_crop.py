import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_size


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class CenterCrop(VisionRandomLayer):
    """Crop the inputs at the center.

    If image size is smaller than output size along any edge, image is padded
    with `padding_value` and then center cropped.

    Args:
        size: Desired output size of the crop. If `size` is an int instead of
            sequence like `(h, w)`, a square crop `(size, size)` is made.
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
        size: typing.Union[typing.Sequence[int], int],
        padding_mode: str = "constant",
        padding_value: float = 0,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(has_generator=False, **kwargs)
        # Check
        available_padding_mode = ("constant", "reflect", "symmetric")
        if padding_mode not in available_padding_mode:
            raise ValueError(
                "Invalid `padding_mode`. Available values are: "
                f"{list(available_padding_mode)}. "
                f"Received: padding_mode={padding_mode}"
            )
        self.size = standardize_size(size)
        self.padding_mode = padding_mode
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.data_format = data_format or backend.image_data_format()

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_shape(self, input_shape):
        images_shape, segmentation_masks_shape = self._get_shape_or_spec(
            input_shape
        )
        images_shape = list(images_shape)
        images_shape[self.h_axis] = self.size[0]
        images_shape[self.w_axis] = self.size[1]
        if segmentation_masks_shape is not None:
            segmentation_masks_shape = list(segmentation_masks_shape)
            segmentation_masks_shape[self.h_axis] = self.size[0]
            segmentation_masks_shape[self.w_axis] = self.size[1]
        return self._set_shape(
            input_shape, images_shape, segmentation_masks_shape
        )

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        images_shape = ops.shape(images)
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        top = ops.numpy.where(
            height < self.size[0], (self.size[0] - height) / 2, 0
        )
        top = ops.cast(ops.numpy.round(top), "int32")
        bottom = ops.numpy.where(
            height < self.size[0], self.size[0] - height - top, 0
        )
        left = ops.numpy.where(
            width < self.size[1], (self.size[1] - width) / 2, 0
        )
        left = ops.cast(ops.numpy.round(left), "int32")
        right = ops.numpy.where(
            width < self.size[1], self.size[1] - width - left, 0
        )
        return dict(top=top, bottom=bottom, left=left, right=right)

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        images_shape = ops.shape(images)
        ori_height = images_shape[self.h_axis]
        ori_width = images_shape[self.w_axis]

        # Pad
        top = transformations["top"]
        bottom = transformations["bottom"]
        left = transformations["left"]
        right = transformations["right"]
        images = self.image_backend.pad(
            images,
            self.padding_mode,
            top,
            bottom,
            left,
            right,
            self.padding_value,
            self.data_format,
        )

        # Center crop
        offset_height = ops.numpy.floor_divide(
            (ori_height + top + bottom - self.size[0]), 2
        )
        offset_width = ops.numpy.floor_divide(
            (ori_width + left + right - self.size[1]), 2
        )
        images = self.image_backend.crop(
            images, offset_height, offset_width, self.size[0], self.size[1]
        )
        return ops.cast(images, self.image_dtype)

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

        x1, y1, x2, y2 = ops.numpy.split(bounding_boxes["boxes"], 4, axis=-1)

        # Get pad and offset
        pad_top = ops.cast(transformations["top"], dtype="float32")
        pad_left = ops.cast(transformations["left"], dtype="float32")
        pad_bottom = ops.cast(transformations["bottom"], dtype="float32")
        pad_right = ops.cast(transformations["right"], dtype="float32")
        offset_height = ops.numpy.floor_divide(
            (height + pad_top + pad_bottom - self.size[0]), 2
        )
        offset_width = ops.numpy.floor_divide(
            (width + pad_left + pad_right - self.size[1]), 2
        )
        value_height = pad_top - offset_height
        value_width = pad_left - offset_width
        x1 = x1 + value_width
        y1 = y1 + value_height
        x2 = x2 + value_width
        y2 = y2 + value_height
        outputs = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
        bounding_boxes = self.bbox_backend.clip_to_images(
            bounding_boxes,
            height=self.size[0],
            width=self.size[1],
            format="xyxy",
        )
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=self.size[0],
            width=self.size[1],
            dtype=self.bounding_box_dtype,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        images_shape = ops.shape(segmentation_masks)
        ori_height = images_shape[self.h_axis]
        ori_width = images_shape[self.w_axis]

        # Pad
        top = transformations["top"]
        bottom = transformations["bottom"]
        left = transformations["left"]
        right = transformations["right"]
        segmentation_masks = self.image_backend.pad(
            segmentation_masks,
            "constant",
            top,
            bottom,
            left,
            right,
            -1,
            self.data_format,
        )

        # Center crop
        offset_height = ops.numpy.floor_divide(
            (ori_height + top + bottom - self.size[0]), 2
        )
        offset_width = ops.numpy.floor_divide(
            (ori_width + left + right - self.size[1]), 2
        )
        segmentation_masks = self.image_backend.crop(
            segmentation_masks,
            offset_height,
            offset_width,
            self.size[0],
            self.size[1],
        )
        return segmentation_masks

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "padding_mode": self.padding_mode,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config
