import typing
from collections.abc import Sequence

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer
from keras_aug._src.utils.argument_validation import standardize_size
from keras_aug._src.utils.argument_validation import standardize_value_range


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomResizedCrop(VisionRandomLayer):
    """Crop a random portion of the images and resize it to a given size.

    Args:
        size: Desired output size. If `size` is a sequence like `(h, w)`,
            output size will be matched to this. If `size` is an int,  a square
            output size `(size, size)` is made.
        scale: A tuple of floats specifying the lower and upper bounds for the
            random area of the crop, before resizing. The scale is defined with
            respect to the area of the original image.
        ratio: A tuple of floats specifying the lower and upper bounds for the
            random aspect ratio of the crop, before resizing.
        interpolation: The interpolation mode. Available values are:
            `"nearest", "bilinear", "bicubic"`. Defaults to `"bilinear"`.
        antialias: Whether to apply antialiasing. It only affects bilinear and
            bicubic modes. Defaults to `True`.
        along_long_edge: Whether to resize the long edge instead of the short
            edge. It only affects when `size` is a single integer. Defaults to
            `False`.
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
        scale: typing.Tuple[float, float] = (0.08, 1.0),
        ratio: typing.Tuple[float, float] = (3 / 4, 4 / 3),
        interpolation: str = "bilinear",
        antialias: bool = True,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(has_generator=False, **kwargs)
        # Check
        if not isinstance(size, (int, Sequence)):
            raise ValueError(
                "`size` must be an integer or a sequence. "
                f"Received: size={size} (of type {type(size)})"
            )
        if isinstance(size, Sequence):
            if len(size) not in (1, 2):
                raise ValueError(
                    "If `size` is a sequence, the length must be 2. "
                    f"Received: size={size}"
                )

        self.size = standardize_size(size)
        self.scale = standardize_value_range(scale)
        self.ratio = standardize_value_range(ratio)
        self.interpolation = interpolation
        self.antialias = antialias
        self.bounding_box_format = bounding_box_format
        self.data_format = data_format or backend.image_data_format()

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_spec(self, inputs):
        images, segmentation_masks = self._get_shape_or_spec(inputs)
        images_shape = list(images.shape)
        h, w = images_shape[self.h_axis], images_shape[self.w_axis]

        # Compute outputs
        if h is not None and w is not None:
            new_h, new_w = self._compute_resized_output_size(
                h, w, self.size, self.along_long_edge
            )
        else:
            new_h, new_w = None, None
        images_shape = list(images_shape)
        images_shape[self.h_axis] = new_h
        images_shape[self.w_axis] = new_w
        images.shape = tuple(images_shape)
        if segmentation_masks is not None:
            segmentation_masks_shape = list(segmentation_masks)
            segmentation_masks_shape[self.h_axis] = new_h
            segmentation_masks_shape[self.w_axis] = new_w
            segmentation_masks.shape = tuple(segmentation_masks_shape)

        return self._set_spec(inputs, images, segmentation_masks)

    def compute_output_shape(self, input_shape):
        images_shape, segmentation_masks_shape = self._get_shape_or_spec(
            input_shape
        )
        h, w = images_shape[self.h_axis], images_shape[self.w_axis]

        # Compute output shape
        if h is not None and w is not None:
            new_h, new_w = self._compute_resized_output_size(
                h, w, self.size, self.along_long_edge
            )
        else:
            new_h, new_w = None, None
        images_shape = list(images_shape)
        images_shape[self.h_axis] = new_h
        images_shape[self.w_axis] = new_w
        if segmentation_masks_shape is not None:
            segmentation_masks_shape = list(segmentation_masks_shape)
            segmentation_masks_shape[self.h_axis] = new_h
            segmentation_masks_shape[self.w_axis] = new_w

        return self._set_shape(
            input_shape, images_shape, segmentation_masks_shape
        )

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        ops = self.backend
        images_shape = ops.shape(images)
        h, w = images_shape[self.h_axis], images_shape[self.w_axis]
        new_h, new_w = self._compute_resized_output_size(
            h, w, self.size, self.along_long_edge
        )
        output_size = ops.numpy.stack([new_h, new_w])
        return output_size

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        output_size = transformations
        images = ops.image.resize(
            images,
            (output_size[0], output_size[1]),
            interpolation=self.interpolation,
            antialias=self.antialias,
            data_format=self.data_format,
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
        output_size = transformations

        # Resize
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="rel_xyxy",
            height=ori_height,
            width=ori_width,
        )
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source="rel_xyxy",
            target=self.bounding_box_format,
            height=ops.cast(output_size[0], "float32"),
            width=ops.cast(output_size[1], "float32"),
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        output_size = transformations
        segmentation_masks = ops.image.resize(
            segmentation_masks,
            (output_size[0], output_size[1]),
            interpolation="nearest",
            antialias=self.antialias,
            data_format=self.data_format,
        )
        return ops.cast(segmentation_masks, self.compute_dtype)

    def _compute_resized_output_size(
        self, height, width, size, along_long_edge=False
    ):
        ops = self.backend
        if isinstance(size, int):
            size = [size]
        if len(size) == 1 and along_long_edge is False:
            new_short = size[0]
            new_long = ops.cast(
                ops.numpy.where(
                    height < width,
                    new_short * width / height,
                    new_short * height / width,
                ),
                "int32",
            )
            new_h = ops.numpy.where(height < width, new_short, new_long)
            new_w = ops.numpy.where(width < height, new_short, new_long)
        elif len(size) == 1 and along_long_edge is True:
            new_long = size[0]
            new_short = ops.cast(
                ops.numpy.where(
                    height > width,
                    new_long * width / height,
                    new_long * height / width,
                ),
                "int32",
            )
            new_h = ops.numpy.where(height < width, new_short, new_long)
            new_w = ops.numpy.where(width < height, new_short, new_long)
        elif len(size) == 2:
            new_h, new_w = size
        else:
            raise ValueError
        return new_h, new_w

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "size": self.size,
                "interpolation": self.interpolation,
                "antialias": self.antialias,
                "along_long_edge": self.along_long_edge,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config
