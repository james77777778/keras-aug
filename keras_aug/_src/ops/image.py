from keras.src.utils.backend_utils import in_tf_graph

from keras_aug._src.backend.image import ImageBackend
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def transform_dtype(images, from_dtype, to_dtype):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).transform_dtype(images, from_dtype, to_dtype)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def crop(images, top, left, height, width, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).crop(
        images, top, left, height, width, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def pad(
    images, mode, top, bottom, left, right, constant_value=0, data_format=None
):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).pad(
        images,
        mode,
        top,
        bottom,
        left,
        right,
        constant_value=constant_value,
        data_format=data_format,
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_brightness(images, factor):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).adjust_brightness(images, factor)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_contrast(images, factor, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).adjust_contrast(
        images, factor, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_saturation(images, factor, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).adjust_saturation(
        images, factor, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_hue(images, factor, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).adjust_hue(
        images, factor, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def affine(
    images,
    angle,
    translate_x,
    translate_y,
    scale,
    shear_x,
    shear_y,
    center_x=None,
    center_y=None,
    interpolation="bilinear",
    padding_mode="constant",
    padding_value=0,
    data_format=None,
):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).affine(
        images,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        center_x=center_x,
        center_y=center_y,
        interpolation=interpolation,
        padding_mode=padding_mode,
        padding_value=padding_value,
        data_format=data_format,
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def auto_contrast(images, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).auto_contrast(images, data_format=data_format)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def blend(images1, images2, factor):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).blend(images1, images2, factor)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def equalize(images, bins=256, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).equalize(
        images, bins=bins, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def guassian_blur(images, kernel_size, sigma, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).guassian_blur(
        images, kernel_size, sigma, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def rgb_to_grayscale(images, num_channels=3, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).rgb_to_grayscale(
        images, num_channels=num_channels, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def invert(images):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).invert(images)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def posterize(images, bits):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).posterize(images, bits)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def sharpen(images, factor, data_format=None):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).sharpen(
        images, factor, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def solarize(images, threshold):
    backend = "tensorflow" if in_tf_graph() else None
    return ImageBackend(backend).solarize(images, threshold)
