import math

import tensorflow as tf
from keras_cv import bounding_box
from keras_cv.layers import VectorizedBaseImageAugmentationLayer
from keras_cv.utils import preprocessing as preprocessing_utils
from tensorflow import keras

from keras_aug.utils import augmentation_utils
from keras_aug.utils.augmentation_utils import H_AXIS
from keras_aug.utils.augmentation_utils import W_AXIS


@keras.utils.register_keras_serializable(package="keras_aug")
class RandomAffine(VectorizedBaseImageAugmentationLayer):
    """A preprocessing layer which randomly affines transformation of the images
    keeping center invariant.

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format
    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format

    Args:
        TODO: docstring
        rotation_factor: a float represented as fraction of 2 Pi, or a tuple of
            size 2 representing lower and upper bound for rotating clockwise and
            counter-clockwise. A positive values means rotating counter
            clock-wise, while a negative value means clock-wise. When
            represented as a single float, this value is used for both the upper
            and lower bound. For instance, `factor=(-0.2, 0.3)` results in an
            output rotation by a random amount in the range
            `[-20% * 2pi, 30% * 2pi]`. `factor=0.2` results in an output
            rotating by a random amount in the range `[-20% * 2pi, 20% * 2pi]`.
        translation_height_factor: a float represented as fraction of value, or
            a tuple of size 2 representing lower and upper bound for shifting
            vertically. A negative value means shifting image up, while a
            positive value means shifting image down. When represented as a
            single positive float, this value is used for both the upper and
            lower bound. For instance, `height_factor=(-0.2, 0.3)` results in an
            output shifted by a random amount in the range `[-20%, +30%]`.
            `height_factor=0.2` results in an output height shifted by a random
            amount in the range `[-20%, +20%]`.
        translation_width_factor: a float represented as fraction of value, or a
            tuple of size 2 representing lower and upper bound for shifting
            horizontally. A negative value means shifting image left, while a
            positive value means shifting image right. When represented as a
            single positive float, this value is used for both the upper and
            lower bound. For instance, `width_factor=(-0.2, 0.3)` results in an
            output shifted left by 20%, and shifted right by 30%.
            `width_factor=0.2` results in an output height shifted left or right
            by 20%.
        zoom_height_factor: a float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound for zooming vertically.
            When represented as a single float, this value is used for both the
            upper and lower bound. A positive value means zooming out, while a
            negative value means zooming in. For instance,
            `height_factor=(0.2, 0.3)` result in an output zoomed out by a
            random amount in the range `[+20%, +30%]`.
            `height_factor=(-0.3, -0.2)` result in an output zoomed in by a
            random amount in the range `[-30%, -20%]`.
        zoom_width_factor: a float represented as fraction of value, or a tuple
            of size 2 representing lower and upper bound for zooming
            horizontally. When represented as a single float, this value is used
            for both the upper and lower bound. For instance,
            `width_factor=(0.2, 0.3)` result in an output zooming out between
            20% to 30%. `width_factor=(-0.3, -0.2)` result in an output zooming
            in between 20% to 30%. Defaults to `None`, i.e., zooming vertical
            and horizontal directions by preserving the aspect ratio. If
            height_factor=0 and width_factor=None, it would result in images
            with no zoom at all.
        shear_height_factor:.
        shear_width_factor:.
        interpolation: Interpolation mode. Supported values: `"nearest"`,
            `"bilinear"`.
        fill_mode: Points outside the boundaries of the input are filled
            according to the given mode
            (one of `{"constant", "reflect", "wrap", "nearest"}`).
            - *reflect*: `(d c b a | a b c d | d c b a)` The input is extended
            by reflecting about the edge of the last pixel.
            - *constant*: `(k k k k | a b c d | k k k k)` The input is extended
            by filling all values beyond the edge with the same constant value
            k = 0.
            - *wrap*: `(a b c d | a b c d | a b c d)` The input is extended by
            wrapping around to the opposite edge.
            - *nearest*: `(a a a a | a b c d | d d d d)` The input is extended
            by the nearest pixel.
        fill_value: a float represents the value to be filled outside the
            boundaries when `fill_mode="constant"`.
        bounding_box_format: The format of bounding boxes of input dataset.
            Refer
            https://github.com/keras-team/keras-cv/blob/master/keras_cv/bounding_box/converters.py
            for more details on supported bounding box formats.
        segmentation_classes: an optional integer with the number of classes in
            the input segmentation mask. Required iff augmenting data with
            sparse (non one-hot) segmentation masks. Include the background
            class in this count
            (e.g. for segmenting dog vs background, this should be set to 2).
        seed: Integer. Used to create a random seed.
    """  # noqa: E501

    def __init__(
        self,
        rotation_factor,
        translation_height_factor,
        translation_width_factor,
        zoom_height_factor,
        zoom_width_factor,
        shear_height_factor,
        shear_width_factor,
        interpolation="bilinear",
        fill_mode="constant",
        fill_value=0,
        bounding_box_format=None,
        segmentation_classes=None,
        seed=None,
        **kwargs,
    ):
        super().__init__(seed=seed, **kwargs)
        # rotation
        if isinstance(rotation_factor, (tuple, list)):
            lower = rotation_factor[0] * 2.0 * math.pi
            upper = rotation_factor[1] * 2.0 * math.pi
        else:
            lower = -rotation_factor * 2.0 * math.pi
            upper = rotation_factor * 2.0 * math.pi
        self.rotation_factor_input = rotation_factor
        self.rotation_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-2.0 * math.pi, max_value=2.0 * math.pi
        )
        # translation
        if isinstance(translation_height_factor, (tuple, list)):
            lower = translation_height_factor[0]
            upper = translation_height_factor[1]
        else:
            lower = -translation_height_factor
            upper = translation_height_factor
        self.translation_height_factor_input = translation_height_factor
        self.translation_height_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )
        if isinstance(translation_width_factor, (tuple, list)):
            lower = translation_width_factor[0]
            upper = translation_width_factor[1]
        else:
            lower = -translation_width_factor
            upper = translation_width_factor
        self.translation_width_factor_input = translation_width_factor
        self.translation_width_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )
        # zoom
        if isinstance(zoom_height_factor, (tuple, list)):
            lower = 1.0 + zoom_height_factor[0]
            upper = 1.0 + zoom_height_factor[1]
        else:
            lower = 1.0 - zoom_height_factor
            upper = 1.0 + zoom_height_factor
        self.zoom_height_factor_input = zoom_height_factor
        self.zoom_height_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=0, max_value=None
        )
        if isinstance(zoom_width_factor, (tuple, list)):
            lower = 1.0 + zoom_width_factor[0]
            upper = 1.0 + zoom_width_factor[1]
        else:
            lower = 1.0 - zoom_width_factor
            upper = 1.0 + zoom_width_factor
        self.zoom_width_factor_input = zoom_width_factor
        self.zoom_width_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=0, max_value=None
        )
        # shear
        if isinstance(shear_height_factor, (tuple, list)):
            lower = shear_height_factor[0]
            upper = shear_height_factor[1]
        else:
            lower = -shear_height_factor
            upper = shear_height_factor
        self.shear_height_factor_input = shear_height_factor
        self.shear_height_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )
        if isinstance(shear_width_factor, (tuple, list)):
            lower = shear_width_factor[0]
            upper = shear_width_factor[1]
        else:
            lower = -shear_width_factor
            upper = shear_width_factor
        self.shear_width_factor_input = shear_width_factor
        self.shear_width_factor = preprocessing_utils.parse_factor(
            (lower, upper), min_value=-1, max_value=1
        )

        preprocessing_utils.check_fill_mode_and_interpolation(
            fill_mode, interpolation
        )
        self.interpolation = interpolation
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.bounding_box_format = bounding_box_format
        self.segmentation_classes = segmentation_classes
        self.seed = seed

    def get_random_transformation_batch(
        self, batch_size, images=None, **kwargs
    ):
        angles = self.rotation_factor(shape=(batch_size, 1))
        translation_heights = self.translation_height_factor(
            shape=(batch_size, 1)
        )
        translation_widths = self.translation_width_factor(
            shape=(batch_size, 1)
        )
        translations = tf.concat(
            [translation_widths, translation_heights], axis=1
        )
        zoom_heights = self.zoom_height_factor(shape=(batch_size, 1))
        zoom_widths = self.zoom_width_factor(shape=(batch_size, 1))
        zooms = tf.concat([zoom_widths, zoom_heights], axis=1)
        shear_heights = self.shear_height_factor(shape=(batch_size, 1))
        shear_widths = self.shear_width_factor(shape=(batch_size, 1))
        shears = tf.concat([shear_widths, shear_heights], axis=1)
        return {
            "angles": angles,
            "translations": translations,
            "zooms": zooms,
            "shears": shears,
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
        original_shape = images.shape
        batch_size = tf.shape(images)[0]
        height = tf.cast(tf.shape(images)[H_AXIS], dtype=tf.float32)
        width = tf.cast(tf.shape(images)[W_AXIS], dtype=tf.float32)
        angles = transformations["angles"]
        translations = transformations["translations"]
        zooms = transformations["zooms"]
        shears = transformations["shears"]

        rotation_matrix = augmentation_utils.get_rotation_matrix(
            angles, height, width, to_square=True
        )
        translation_matrix = augmentation_utils.get_translation_matrix(
            translations, height, width, to_square=True
        )
        zoom_matrix = augmentation_utils.get_zoom_matrix(
            zooms, height, width, to_square=True
        )
        shear_matrix = augmentation_utils.get_shear_matrix(
            shears, to_square=True
        )
        combined_matrix = (
            translation_matrix @ shear_matrix @ zoom_matrix @ rotation_matrix
        )
        combined_matrix = tf.reshape(combined_matrix, shape=(batch_size, -1))
        combined_matrix = combined_matrix[:, :-1]

        images = preprocessing_utils.transform(
            images,
            combined_matrix,
            fill_mode=self.fill_mode,
            fill_value=self.fill_value,
            interpolation=self.interpolation,
        )
        images.set_shape(original_shape)
        return images

    def augment_labels(self, labels, transformations, **kwargs):
        return labels

    def augment_bounding_boxes(
        self, bounding_boxes, transformations, raw_images=None, **kwargs
    ):
        if self.bounding_box_format is None:
            raise ValueError(
                "`RandomAffine()` was called with bounding boxes,"
                "but no `bounding_box_format` was specified in the constructor."
                "Please specify a bounding box format in the constructor. i.e."
                "`RandomAffine(bounding_box_format='xyxy')`"
            )
        bounding_boxes = bounding_box.to_dense(bounding_boxes)
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source=self.bounding_box_format,
            target="xyxy",
            images=raw_images,
        )

        bounding_boxes = bounding_box.clip_to_image(
            bounding_boxes,
            bounding_box_format="xyxy",
            images=raw_images,
        )
        # coordinates cannot be float values, it is cast to int32
        bounding_boxes = bounding_box.convert_format(
            bounding_boxes,
            source="xyxy",
            target=self.bounding_box_format,
            dtype=self.compute_dtype,
            images=raw_images,
        )
        return bounding_boxes

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "rotation_factor": self.rotation_factor_input,
                "translation_height_factor": self.translation_height_factor_input,  # noqa: E501
                "translation_width_factor": self.translation_width_factor_input,
                "zoom_height_factor": self.zoom_height_factor_input,
                "zoom_width_factor": self.zoom_width_factor_input,
                "shear_height_factor": self.shear_height_factor_input,
                "shear_width_factor": self.shear_width_factor_input,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "interpolation": self.interpolation,
                "bounding_box_format": self.bounding_box_format,
                "segmentation_classes": self.segmentation_classes,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
