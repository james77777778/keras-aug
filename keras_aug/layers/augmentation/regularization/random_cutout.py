import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.bounding_box.iou import _compute_area
from keras_cv.bounding_box.iou import _compute_intersection
from keras_cv.utils import fill_utils
from tensorflow import keras

from keras_aug.layers.base.vectorized_base_random_layer import (
    VectorizedBaseRandomLayer,
)
from keras_aug.utils import augmentation as augmentation_utils


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomCutout(VectorizedBaseRandomLayer):
    """Randomly cut out rectangles from images and fill them.

    Args:
        height_factor (float|Sequence[float]|keras_aug.FactorSampler): The range
            of the height factor that controls the height of the cutout
            rectangle. When represented as a single float, the factor will be
            picked between ``[0.0, 0.0 + upper]``. ``0.0`` means the rectangle
            will be of size 0% of the image height. ``0.1`` means the rectangle
            will have a size of 10% of the image height.
        width_factor (float|Sequence[float]|keras_aug.FactorSampler): The range
            of the width factor that controls the width of the cutout
            rectangle. When represented as a single float, the factor will be
            picked between ``[0.0, 0.0 + upper]``. ``0.0`` means the rectangle
            will be of size 0% of the image width. ``0.1`` means the rectangle
            will have a size of 10% of the image width.
        fill_mode (str, optional): Pixels inside the cutout rectangle are filled
            according to the given mode. Supported values:
            ``"constant", "gaussian_noise"``. Defaults to ``"constant"``.
        fill_value (int|float, optional): The value to be filled in the cutout
            rectangle when ``fill_mode="constant"``. Defaults to ``0``.
        bbox_removal_threshold (float, optional): The bounding boxes having
            content cut above the threshold will be removed.
            Defaults to ``0.6`` which is applied by ultralytics/yolo series.
        bounding_box_format (str, optional): The format of bounding
            boxes of input dataset. Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        seed (int|float, optional): The random seed. Defaults to ``None``.

    References:
        - `Cutout <https://arxiv.org/abs/1708.04552>`_
        - `KerasCV <https://github.com/keras-team/keras-cv>`_
        - `kaushal2896@Kaggle <https://www.kaggle.com/code/kaushal2896/data-augmentation-tutorial-basic-cutout-mixup>`_
        - `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
    """  # noqa: E501

    def __init__(
        self,
        height_factor,
        width_factor,
        fill_mode="constant",
        fill_value=0.0,
        bbox_removal_threshold=0.6,
        bounding_box_format=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        if isinstance(height_factor, (int, float)):
            height_factor = (0, height_factor)
        self.height_factor = augmentation_utils.parse_factor(
            height_factor, seed=seed
        )
        if isinstance(width_factor, (int, float)):
            width_factor = (0, width_factor)
        self.width_factor = augmentation_utils.parse_factor(
            width_factor, seed=seed
        )
        if fill_mode not in ["gaussian_noise", "constant"]:
            raise ValueError(
                '`fill_mode` should be "gaussian_noise" '
                f'or "constant". Got `fill_mode`={fill_mode}'
            )
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.bbox_removal_threshold = bbox_removal_threshold
        self.bounding_box_format = bounding_box_format
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        heights, widths = augmentation_utils.get_images_shape(
            images, dtype=tf.float32
        )
        center_xs = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        center_ys = self._random_generator.random_uniform(
            shape=(batch_size, 1), minval=0, maxval=1, dtype=tf.float32
        )
        center_xs = tf.cast(tf.round(center_xs * widths), dtype=tf.int32)
        center_ys = tf.cast(tf.round(center_ys * heights), dtype=tf.int32)

        cutout_heights = self.height_factor(shape=(batch_size, 1))
        cutout_widths = self.width_factor(shape=(batch_size, 1))
        cutout_heights = cutout_heights * heights
        cutout_widths = cutout_widths * widths
        cutout_heights = tf.cast(tf.math.ceil(cutout_heights), tf.int32)
        cutout_widths = tf.cast(tf.math.ceil(cutout_widths), tf.int32)
        cutout_heights = tf.minimum(
            cutout_heights, tf.cast(heights, dtype=tf.int32)
        )
        cutout_widths = tf.minimum(
            cutout_widths, tf.cast(widths, dtype=tf.int32)
        )
        return {
            "center_xs": center_xs,
            "center_ys": center_ys,
            "cutout_heights": cutout_heights,
            "cutout_widths": cutout_widths,
        }

    def augment_ragged_image(self, image, transformation, **kwargs):
        image = tf.expand_dims(image, axis=0)
        transformation = augmentation_utils.expand_dict_dims(
            transformation, axis=0
        )
        image = self.augment_images(
            images=image, transformations=transformation, **kwargs
        )
        return tf.squeeze(image, axis=0)

    def augment_images(self, images, transformations, **kwargs):
        images = tf.cast(images, dtype=self.compute_dtype)
        center_xs = transformations["center_xs"]
        center_ys = transformations["center_ys"]
        cutout_heights = transformations["cutout_heights"]
        cutout_widths = transformations["cutout_widths"]
        rectangle_fills = self.compute_rectangle_fills(images)
        images = fill_utils.fill_rectangle(
            images,
            center_xs[..., 0],
            center_ys[..., 0],
            cutout_widths[..., 0],
            cutout_heights[..., 0],
            rectangle_fills,
        )
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomCutout()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomCutout(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
            dtype=tf.float32,
        )
        # construct cutout bounding_boxes
        cutout_boxes = tf.concat(
            [
                transformations["center_xs"],
                transformations["center_ys"],
                transformations["cutout_widths"],
                transformations["cutout_heights"],
            ],
            axis=-1,
        )
        cutout_boxes = tf.expand_dims(cutout_boxes, axis=1)
        cutout_boxes = bounding_box.convert_format(
            cutout_boxes,
            source="center_xywh",
            target="xyxy",
            images=raw_images,
            dtype=tf.float32,
        )

        areas = _compute_area(bounding_boxes["boxes"])
        intersections = tf.squeeze(
            _compute_intersection(bounding_boxes["boxes"], cutout_boxes),
            axis=-1,
        )
        intersection_ratios = tf.math.divide_no_nan(intersections, areas)

        # set classes == -1 if ratios < self.bbox_removal_threshold
        # the bounding_boxes with classes==-1 will be removed by
        # bounding_box.to_ragged() after self.augment_bounding_boxes()
        bounding_boxes["classes"] = tf.where(
            intersection_ratios >= self.bbox_removal_threshold,
            -1.0,
            bounding_boxes["classes"],
        )
        return bounding_boxes

    def compute_rectangle_fills(self, inputs):
        input_shape = tf.shape(inputs)
        if self.fill_mode == "constant":
            fill_value = tf.fill(input_shape, self.fill_value)
            fill_value = tf.cast(fill_value, dtype=self.compute_dtype)
        else:
            # gaussian noise
            fill_value = tf.random.normal(input_shape, dtype=self.compute_dtype)
        return fill_value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "height_factor": self.height_factor,
                "width_factor": self.width_factor,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "bbox_removal_threshold": self.bbox_removal_threshold,
                "bounding_box_format": self.bounding_box_format,
                "seed": self.seed,
            }
        )
        return config
