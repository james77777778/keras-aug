import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_size


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomCrop(VisionRandomLayer):
    """Crop the images at a random location.

    If image size is smaller than output size along any edge, image is padded
    with `padding_value` and then cropped.

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
        super().__init__(**kwargs)
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
        random_generator = self.random_generator
        images_shape = ops.shape(images)
        height, width = images_shape[self.h_axis], images_shape[self.w_axis]

        padded_height = height
        padded_width = width
        cropped_height, cropped_width = self.size

        # Pad parameters
        # self.pad_if_needed always True
        pad_top = 0
        pad_bottom = 0
        pad_left = 0
        pad_right = 0
        height_diff = ops.numpy.maximum(
            ops.numpy.subtract(cropped_height, padded_height), 0
        )
        width_diff = ops.numpy.maximum(
            ops.numpy.subtract(cropped_width, padded_width), 0
        )
        padded_height = ops.numpy.add(padded_height, 2 * height_diff)
        padded_width = ops.numpy.add(padded_width, 2 * width_diff)
        pad_top = ops.numpy.add(pad_top, height_diff)
        pad_bottom = ops.numpy.add(pad_bottom, height_diff)
        pad_left = ops.numpy.add(pad_left, width_diff)
        pad_right = ops.numpy.add(pad_right, width_diff)

        # Crop parameters
        crop_top = ops.random.randint(
            (),
            minval=0,
            maxval=padded_height - cropped_height + 1,
            seed=random_generator,
        )
        crop_left = ops.random.randint(
            (),
            minval=0,
            maxval=padded_width - cropped_width + 1,
            seed=random_generator,
        )
        return dict(
            pad_top=pad_top,
            pad_bottom=pad_bottom,
            pad_left=pad_left,
            pad_right=pad_right,
            crop_top=crop_top,
            crop_left=crop_left,
        )

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend

        # Pad
        pad_top = transformations["pad_top"]
        pad_bottom = transformations["pad_bottom"]
        pad_left = transformations["pad_left"]
        pad_right = transformations["pad_right"]
        images = self.image_backend.pad(
            images,
            self.padding_mode,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            self.padding_value,
            self.data_format,
        )

        # Crop
        crop_top = transformations["crop_top"]
        crop_left = transformations["crop_left"]
        images = self.image_backend.crop(
            images, crop_top, crop_left, self.size[0], self.size[1]
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
        images_shape = ops.shape(raw_images)
        height = images_shape[self.h_axis]
        width = images_shape[self.w_axis]
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            height=height,
            width=width,
        )

        x1, y1, x2, y2 = ops.numpy.split(bounding_boxes["boxes"], 4, axis=-1)

        # Get pad and offset
        pad_top = ops.cast(transformations["pad_top"], dtype="float32")
        pad_left = ops.cast(transformations["pad_left"], dtype="float32")
        crop_top = ops.cast(transformations["crop_top"], dtype="float32")
        crop_left = ops.cast(transformations["crop_left"], dtype="float32")
        value_height = pad_top - crop_top
        value_width = pad_left - crop_left
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
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend

        # Pad
        pad_top = transformations["pad_top"]
        pad_bottom = transformations["pad_bottom"]
        pad_left = transformations["pad_left"]
        pad_right = transformations["pad_right"]
        segmentation_masks = self.image_backend.pad(
            segmentation_masks,
            "constant",
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            -1,
            self.data_format,
        )

        # Center crop
        crop_top = transformations["crop_top"]
        crop_left = transformations["crop_left"]
        segmentation_masks = self.image_backend.crop(
            segmentation_masks,
            crop_top,
            crop_left,
            self.size[0],
            self.size[1],
        )
        return ops.cast(segmentation_masks, self.compute_dtype)

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
