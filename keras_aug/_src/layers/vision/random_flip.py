import typing

import keras
from keras import backend

from keras_aug._src.keras_aug_export import keras_aug_export
from keras_aug._src.layers.base.vision_random_layer import VisionRandomLayer


@keras_aug_export(parent_path=["keras_aug.layers.vision", "keras_aug.layers"])
@keras.saving.register_keras_serializable(package="keras_aug")
class RandomFlip(VisionRandomLayer):
    """Flip the inputs with a given probability.

    Args:
        mode: Can be `"horizontal"`, `"vertical"` and
            `"horizontal_and_vertical"`. Defaults to `"horizontal"`.
        p: A float specifying the probability. Defaults to `0.5`.
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
        mode: str = "horizontal",
        p: float = 0.5,
        bounding_box_format: typing.Optional[str] = None,
        data_format: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(mode, str):
            raise TypeError(
                "`mode` must be a string. "
                f"Received: mode={mode} of type {type(mode)}"
            )
        mode = mode.lower()
        if mode not in ("horizontal", "vertical", "horizontal_and_vertical"):
            raise ValueError(
                "`mode` must be either 'horizontal' or 'vertical'. "
                f"Received: mode={mode}"
            )
        self.mode = mode
        self.p = float(p)
        self.bounding_box_format = bounding_box_format
        self.data_format = data_format or backend.image_data_format()

        if self.data_format == "channels_last":
            self.h_axis, self.w_axis = -3, -2
        else:
            self.h_axis, self.w_axis = -2, -1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_params(self, batch_size, images=None, **kwargs):
        ops = self.backend
        random_generator = self.random_generator
        if "horizontal" in self.mode:
            p_horizontal = ops.random.uniform(
                [batch_size], seed=random_generator
            )
        else:
            p_horizontal = ops.numpy.ones([batch_size], "float32")
        if "vertical" in self.mode:
            p_vertical = ops.random.uniform([batch_size], seed=random_generator)
        else:
            p_vertical = ops.numpy.ones([batch_size], "float32")
        return dict(p_horizontal=p_horizontal, p_vertical=p_vertical)

    def augment_images(self, images, transformations, **kwargs):
        ops = self.backend
        p_horizontal = transformations["p_horizontal"]
        p_vertical = transformations["p_vertical"]
        prob_horizontal = ops.numpy.expand_dims(
            p_horizontal < self.p, axis=[1, 2, 3]
        )
        prob_vertical = ops.numpy.expand_dims(
            p_vertical < self.p, axis=[1, 2, 3]
        )

        # Horizontal
        images = ops.numpy.where(
            prob_horizontal,
            ops.numpy.flip(images, axis=self.w_axis),
            images,
        )
        # Vertical
        images = ops.numpy.where(
            prob_vertical,
            ops.numpy.flip(images, axis=self.h_axis),
            images,
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
        images_shape = ops.shape(images)
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

        p_horizontal = transformations["p_horizontal"]
        p_vertical = transformations["p_vertical"]
        prob_horizontal = ops.numpy.expand_dims(
            p_horizontal < self.p, axis=[1, 2]
        )
        prob_vertical = ops.numpy.expand_dims(p_vertical < self.p, axis=[1, 2])

        # Horizontal
        _x1 = ops.numpy.where(
            prob_horizontal, ops.numpy.subtract(width, x2), x1
        )
        _x2 = ops.numpy.where(
            prob_horizontal, ops.numpy.subtract(width, x1), x2
        )
        # Vertical
        _y1 = ops.numpy.where(prob_vertical, ops.numpy.subtract(width, y2), y1)
        _y2 = ops.numpy.where(prob_vertical, ops.numpy.subtract(width, y1), y2)
        outputs = ops.numpy.concatenate([_x1, _y1, _x2, _y2], axis=-1)

        bounding_boxes = bounding_boxes.copy()
        bounding_boxes["boxes"] = outputs
        bounding_boxes = self.bbox_backend.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            height=height,
            width=width,
            dtype=self.compute_dtype,
        )
        return bounding_boxes

    def augment_segmentation_masks(
        self, segmentation_masks, transformations, **kwargs
    ):
        ops = self.backend
        p_horizontal = transformations["p_horizontal"]
        p_vertical = transformations["p_vertical"]
        prob_horizontal = ops.numpy.expand_dims(
            p_horizontal < self.p, axis=[1, 2, 3]
        )
        prob_vertical = ops.numpy.expand_dims(
            p_vertical < self.p, axis=[1, 2, 3]
        )

        # Horizontal
        segmentation_masks = ops.numpy.where(
            prob_horizontal,
            ops.numpy.flip(segmentation_masks, axis=self.w_axis),
            segmentation_masks,
        )
        # Vertical
        segmentation_masks = ops.numpy.where(
            prob_vertical,
            ops.numpy.flip(segmentation_masks, axis=self.h_axis),
            segmentation_masks,
        )
        return ops.cast(segmentation_masks, self.compute_dtype)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "mode": self.mode,
                "p": self.p,
                "bounding_box_format": self.bounding_box_format,
            }
        )
        return config
