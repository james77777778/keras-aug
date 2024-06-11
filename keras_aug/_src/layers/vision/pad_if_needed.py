import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_size


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class PadIfNeeded(VisionRandomLayer):
    """Pad the images to the given size.

    Args:
        size: Desired output size. If `size` is a sequence like `(h, w)`,
            output size will be matched to this. If `size` is an int, a square
            size (size, size) is made.
        padding_mode: The mode of the padding. Available values:
            `"constant", "edge", "reflect", "symmetric"`. Defaults to
            `"constant"`.
        padding_position: The position of the padding. Available values:
            `"border", "top_left", "top_right", "bottom_left", "bottom_right".`
            Defaults to `"border"`.
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
        padding_position: str = "border",
        padding_value: float = 0,
        bounding_box_format=None,
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
        available_padding_position = (
            "border",
            "top_left",
            "top_right",
            "bottom_left",
            "bottom_right",
        )
        if padding_position not in available_padding_position:
            raise ValueError(
                "Invalid `padding_position`. Available values are: "
                f"{list(available_padding_position)}. "
                f"Received: padding_position={padding_position}"
            )

        self.size = standardize_size(size)
        self.padding_mode = padding_mode
        self.padding_position = padding_position
        self.padding_value = padding_value
        self.bounding_box_format = bounding_box_format
        self.data_format = data_format or backend.image_data_format()

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_shape(self, input_shape):
        # Get original h, w
        if not isinstance(input_shape, dict):
            h, w = input_shape[self.h_axis], input_shape[self.w_axis]
        else:
            images_shape = list(input_shape[self.IMAGES])
            h, w = images_shape[self.h_axis], images_shape[self.w_axis]

        # Compute new_h and new_w
        new_h, new_w = None, None
        if h is not None:
            new_h = self.size[0] if self.size[0] > h else h
        if w is not None:
            new_w = self.size[1] if self.size[1] > w else w

        # Update
        if not isinstance(input_shape, dict):
            output_shape = list(input_shape)
            output_shape[self.h_axis] = new_h
            output_shape[self.w_axis] = new_w
        else:
            output_shape = input_shape.copy()
            output_shape[self.IMAGES][self.h_axis] = new_h
            output_shape[self.IMAGES][self.w_axis] = new_w
        return output_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        images_shape = ops.shape(images)
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        pad_h = ops.numpy.where(self.size[0] > height, self.size[0] - height, 0)
        pad_w = ops.numpy.where(self.size[1] > width, self.size[1] - width, 0)
        pad_h = ops.cast(pad_h, "int32")
        pad_w = ops.cast(pad_w, "int32")

        if self.padding_position == "border":
            pad_top = ops.cast(ops.cast(pad_h, "float32") / 2, "int32")
            pad_bottom = pad_h - pad_top
            pad_left = ops.cast(ops.cast(pad_w, "float32") / 2, "int32")
            pad_right = pad_w - pad_left
        elif self.padding_position == "top_left":
            pad_top = pad_h
            pad_bottom = 0
            pad_left = pad_w
            pad_right = 0
        elif self.padding_position == "top_right":
            pad_top = pad_h
            pad_bottom = 0
            pad_left = 0
            pad_right = pad_w
        elif self.padding_position == "bottom_left":
            pad_top = 0
            pad_bottom = pad_h
            pad_left = pad_w
            pad_right = 0
        elif self.padding_position == "bottom_right":
            pad_top = 0
            pad_bottom = pad_h
            pad_left = 0
            pad_right = pad_w
        else:
            raise NotImplementedError

        return {
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend

        # Pad
        pad_top = transformations["pad_top"]
        pad_bottom = transformations["pad_bottom"]
        pad_left = transformations["pad_left"]
        pad_right = transformations["pad_right"]
        if self._backend.name == "torch":  # Workaround for torch
            pad_top = int(pad_top)
            pad_bottom = int(pad_bottom)
            pad_left = int(pad_left)
            pad_right = int(pad_right)
        pad_width = [[pad_top, pad_bottom], [pad_left, pad_right]]
        if self.data_format == "channels_last":
            pad_width = pad_width + [[0, 0]]
        else:
            pad_width = [[0, 0]] + pad_width
        pad_width = [[0, 0]] + pad_width
        images = ops.numpy.pad(
            images,
            pad_width,
            self.padding_mode,
            self.padding_value if self.padding_mode == "constant" else None,
        )
        return ops.cast(images, self.compute_dtype)

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

        ori_images_shape = ops.shape(raw_images)
        ori_height = ori_images_shape[self.h_axis]
        ori_width = ori_images_shape[self.w_axis]

        images_shape = ops.shape(images)
        height = images_shape[self.h_axis]
        width = images_shape[self.w_axis]

        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=ori_height,
            width=ori_width,
        )

        x1, y1, x2, y2 = ops.numpy.split(bounding_boxes["boxes"], 4, axis=-1)

        pad_top = ops.cast(transformations["pad_top"], dtype="float32")
        pad_left = ops.cast(transformations["pad_left"], dtype="float32")
        x1 = x1 + pad_left
        y1 = y1 + pad_top
        x2 = x2 + pad_left
        y2 = y2 + pad_top
        outputs = ops.numpy.concatenate([x1, y1, x2, y2], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=height,
            width=width,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        images_shape = ops.shape(segmentation_masks)

        # Pad
        pad_top = transformations["pad_top"]
        pad_bottom = transformations["pad_bottom"]
        pad_left = transformations["pad_left"]
        pad_right = transformations["pad_right"]
        if self._backend.name == "torch":  # Workaround for torch
            pad_top = int(pad_top)
            pad_bottom = int(pad_bottom)
            pad_left = int(pad_left)
            pad_right = int(pad_right)
        pad_width = [[pad_top, pad_bottom], [pad_left, pad_right]]
        if self.data_format == "channels_last":
            pad_width = pad_width + [[0, 0]]
        else:
            pad_width = [[0, 0]] + pad_width
        if len(images_shape) == 4:
            pad_width = [[0, 0]] + pad_width
        segmentation_masks = ops.numpy.pad(
            segmentation_masks, pad_width, "constant", -1
        )
        return ops.cast(segmentation_masks, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "padding_mode": self.padding_mode,
                "padding_position": self.padding_position,
                "padding_value": self.padding_value,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config
