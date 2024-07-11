from keras_aug._src.backend.image import ImageBackend
from keras_aug._src.keras_aug_export import keras_aug_export


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def transform_dtype(images, dtype):
    return ImageBackend().transform_dtype(images, dtype)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def crop(images, top, left, height, width, data_format=None):
    return ImageBackend().crop(
        images, top, left, height, width, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def pad(
    images, mode, top, bottom, left, right, constant_value=0, data_format=None
):
    return ImageBackend().pad(
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
    return ImageBackend().adjust_brightness(images, factor)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_contrast(images, factor, data_format=None):
    return ImageBackend().adjust_contrast(
        images, factor, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_saturation(images, factor, data_format=None):
    return ImageBackend().adjust_saturation(
        images, factor, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def adjust_hue(images, factor, data_format=None):
    return ImageBackend().adjust_hue(images, factor, data_format=data_format)


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
    return ImageBackend().affine(
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
    return ImageBackend().auto_contrast(images, data_format=data_format)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def blend(images1, images2, factor):
    return ImageBackend().blend(images1, images2, factor)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def equalize(images, bins=256, data_format=None):
    return ImageBackend().equalize(images, bins=bins, data_format=data_format)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def guassian_blur(images, kernel_size, sigma, data_format=None):
    return ImageBackend().guassian_blur(
        images, kernel_size, sigma, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def rgb_to_grayscale(images, num_channels=3, data_format=None):
    return ImageBackend().rgb_to_grayscale(
        images, num_channels=num_channels, data_format=data_format
    )


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def invert(images):
    return ImageBackend().invert(images)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def posterize(images, bits):
    return ImageBackend().posterize(images, bits)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def sharpen(images, factor, data_format=None):
    return ImageBackend().sharpen(images, factor, data_format=data_format)


@keras_aug_export(parent_path=["keras_aug.ops.image"])
def solarize(images, threshold):
    return ImageBackend().solarize(images, threshold)
