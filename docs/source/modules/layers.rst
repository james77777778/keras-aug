API - Layers
=========================

.. automodule:: keras_aug.layers

Table of Layers
****************

**Augmentation 2D**

----

.. autosummary::
    :nosignatures:

    RandAugment
    RandomAffine
    RandomCrop
    RandomCropAndResize
    RandomFlip
    RandomRotate
    RandomZoomAndCrop
    ChannelShuffle
    RandomBlur
    RandomBrightnessContrast
    RandomChannelShift
    RandomCLAHE
    RandomColorJitter
    RandomGamma
    RandomGaussianBlur
    RandomHSV
    RandomJpegQuality
    RandomPosterize
    RandomSharpness
    RandomSolarize
    CutMix
    MixUp
    MosaicYOLOV8
    RandomChannelDropout
    RandomCutout
    RandomErase
    RandomGridMask
    RandomApply

**Preprocessing 2D**

----

.. autosummary::
    :nosignatures:

    CenterCrop
    PadIfNeeded
    Resize
    ResizeAndCrop
    ResizeAndPad
    ResizeByLongestSide
    ResizeBySmallestSide
    AutoContrast
    Equalize
    Grayscale
    Invert
    Normalize
    Rescale
    Identity

**Base 2D**

----

.. autosummary::
    :nosignatures:

    VectorizedBaseRandomLayer

.. -----------------------------------------------------------
..                        2D
.. -----------------------------------------------------------

Augmentation 2D
****************

**Auto**

----

.. autoclass:: RandAugment(value_range, augmentations_per_image=2, magnitude=10, magnitude_stddev=0, cutout_multiplier=60.0 / 331.0, translation_multiplier=150.0/331.0, use_geometry=True, interpolation="nearest", fill_mode="reflect", fill_value=0, exclude_ops=None, bounding_box_format=None, seed=None, **kwargs)

**Geometry**

----

.. autoclass:: RandomAffine(rotation_factor, translation_height_factor, translation_width_factor, zoom_height_factor, zoom_width_factor, shear_height_factor, shear_width_factor, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomCrop(height, width, interpolation="bilinear", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomCropAndResize(height, width, crop_area_factor, aspect_ratio_factor, interpolation="bilinear", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomFlip(mode="horizontal", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomRotate(factor, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomZoomAndCrop(height, width, scale_factor, crop_height=None, crop_width=None, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)

**Intensity**

----

.. autoclass:: ChannelShuffle(groups=3, seed=None, **kwargs)
.. autoclass:: RandomBlur(factor, seed=None, **kwargs)
.. autoclass:: RandomBrightnessContrast(value_range, brightness_factor, contrast_factor, seed=None, **kwargs)
.. autoclass:: RandomChannelShift(value_range, factor, channels=3, seed=None, **kwargs)
.. autoclass:: RandomCLAHE(value_range, factor=(1, 4), tile_grid_size=(8, 8), seed=None, **kwargs)
.. autoclass:: RandomColorJitter(value_range, brightness_factor, contrast_factor, saturation_factor, hue_factor, seed=None, **kwargs)
.. autoclass:: RandomGamma(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomGaussianBlur(kernel_size, factor, seed=None, **kwargs)
.. autoclass:: RandomHSV(value_range, hue_factor, saturation_factor, value_factor, seed=None, **kwargs)
.. autoclass:: RandomJpegQuality(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomPosterize(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomSharpness(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomSolarize(value_range, threshold_factor, addition_factor=0, seed=None, **kwargs)

**Mix**

----

.. autoclass:: CutMix(alpha=1.0, seed=None, **kwargs)
.. autoclass:: MixUp(alpha=0.2, seed=None, **kwargs)
.. autoclass:: MosaicYOLOV8(height, width, offset=(0.25, 0.75), seed=None, **kwargs)

**Regularization**

----

.. autoclass:: RandomChannelDropout(factor=(0, 2), fill_value=0, seed=None, **kwargs)
.. autoclass:: RandomCutout(height_factor, width_factor, fill_mode="constant", fill_value=0, seed=None, **kwargs)
.. autoclass:: RandomErase(area_factor=(0.02, 0.4), aspect_ratio_factor=(0.3, 1.0 / 0.3), fill_mode="constant", fill_value=(125, 123, 114), seed=None, **kwargs)
.. autoclass:: RandomGridMask(size_factor=(96.0 / 224.0, 224.0 / 224.0), ratio_factor=(0.6, 0.6), rotation_factor=(-180, 180), fill_mode="constant", fill_value=0.0, seed=None, **kwargs)

**Utility**

----

.. autoclass:: RandomApply(layer, rate=0.5, seed=None, **kwargs)

Preprocessing 2D
****************

**Geometry**

----

.. autoclass:: CenterCrop(height, width, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: PadIfNeeded(min_height=None, min_width=None, height_divisor=None, width_divisor=None, position="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: Resize(height, width, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeAndCrop(height, width, interpolation="bilinear", antialias=False, postion="center", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeAndPad(height, width, interpolation="bilinear", antialias=False, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeByLongestSide(max_size, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeBySmallestSide(min_size, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)

**Intensity**

----

.. autoclass:: AutoContrast(value_range, **kwargs)
.. autoclass:: Equalize(value_range, bins=256, **kwargs)
.. autoclass:: Grayscale(output_channels=3, **kwargs)
.. autoclass:: Invert(value_range, **kwargs)
.. autoclass:: Normalize(value_range, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs)
.. autoclass:: Rescale(scale, offset=0.0)

**Utility**

----

.. autoclass:: Identity(**kwargs)

Base 2D
****************

.. autoclass:: VectorizedBaseRandomLayer(seed=None, **kwargs)

.. -----------------------------------------------------------
..                        3D
.. -----------------------------------------------------------

Base 3D
****************

WIP