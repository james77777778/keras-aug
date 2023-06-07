API - Layers
=========================

.. automodule:: keras_aug.layers

Table of Layers
****************

**Augmentation 2D**

----

.. autosummary::
    :nosignatures:

    AugMix
    RandAugment
    TrivialAugmentWide
    RandomAffine
    RandomCrop
    RandomCropAndResize
    RandomFlip
    RandomResize
    RandomRotate
    RandomZoomAndCrop
    ChannelShuffle
    RandomBlur
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
    Mosaic
    RandomChannelDropout
    RandomCutout
    RandomErase
    RandomGridMask
    RandomApply
    RandomChoice
    RepeatedAugment

**Preprocessing 2D**

----

.. autosummary::
    :nosignatures:

    CenterCrop
    PadIfNeeded
    Resize
    AutoContrast
    Equalize
    Grayscale
    Invert
    Normalize
    Rescale
    Identity
    SanitizeBoundingBox

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

.. autoclass:: AugMix(value_range, severity=[0.01, 0.3], num_chains=3, chain_depth=[1, 3], alpha=1.0, seed=None, **kwargs)
.. autoclass:: RandAugment(value_range, augmentations_per_image=2, magnitude=10, magnitude_stddev=0, translation_multiplier=150.0/331.0, use_geometry=True, interpolation="nearest", fill_mode="reflect", fill_value=0, exclude_ops=None, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: TrivialAugmentWide(value_range, use_geometry=True, interpolation="nearest", fill_mode="reflect", fill_value=0, exclude_ops=None, bounding_box_format=None, seed=None, **kwargs)

**Geometry**

----

.. autoclass:: RandomAffine(rotation_factor=None, translation_height_factor=None, translation_width_factor=None, zoom_height_factor=None, zoom_width_factor=None, shear_height_factor=None, shear_width_factor=None, same_zoom_factor=False, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, bounding_box_min_area_ratio=None, bounding_box_max_aspect_ratio=None, seed=None, **kwargs)
.. autoclass:: RandomCrop(height, width, interpolation="bilinear", bounding_box_format=None, bounding_box_min_area_ratio=None, bounding_box_max_aspect_ratio=None, seed=None, **kwargs)
.. autoclass:: RandomCropAndResize(height, width, crop_area_factor, aspect_ratio_factor, interpolation="bilinear", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomFlip(mode="horizontal", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomResize(heights, widths=None, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomRotate(factor, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomZoomAndCrop(height, width, scale_factor, crop_height=None, crop_width=None, interpolation="bilinear", antialias=False, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)

**Intensity**

----

.. autoclass:: ChannelShuffle(groups=3, seed=None, **kwargs)
.. autoclass:: RandomBlur(factor, seed=None, **kwargs)
.. autoclass:: RandomChannelShift(value_range, factor, channels=3, seed=None, **kwargs)
.. autoclass:: RandomCLAHE(value_range, factor=(4, 4), tile_grid_size=(8, 8), seed=None, **kwargs)
.. autoclass:: RandomColorJitter(value_range, brightness_factor=None, contrast_factor=None, saturation_factor=None, hue_factor=None, seed=None, **kwargs)
.. autoclass:: RandomGamma(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomGaussianBlur(kernel_size, factor, seed=None, **kwargs)
.. autoclass:: RandomHSV(value_range, hue_factor=None, saturation_factor=None, value_factor=None, seed=None, **kwargs)
.. autoclass:: RandomJpegQuality(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomPosterize(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomSharpness(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomSolarize(value_range, threshold_factor, addition_factor=0, seed=None, **kwargs)

**Mix**

----

.. autoclass:: CutMix(alpha=1.0, seed=None, **kwargs)
.. autoclass:: MixUp(alpha=0.2, seed=None, **kwargs)
.. autoclass:: Mosaic(height, width, offset=(0.25, 0.75), fill_value=0, bounding_box_format=None, seed=None, **kwargs)

**Regularization**

----

.. autoclass:: RandomChannelDropout(factor=(0, 2), fill_value=0, seed=None, **kwargs)
.. autoclass:: RandomCutout(height_factor, width_factor, fill_mode="constant", fill_value=0, bbox_removal_threshold=0.6, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomErase(area_factor=(0.02, 0.4), aspect_ratio_factor=(0.3, 1.0 / 0.3), fill_mode="constant", fill_value=(125, 123, 114), seed=None, **kwargs)
.. autoclass:: RandomGridMask(size_factor=(96.0 / 224.0, 224.0 / 224.0), ratio_factor=(0.6, 0.6), rotation_factor=(-180, 180), fill_mode="constant", fill_value=0.0, seed=None, **kwargs)

**Utility**

----

.. autoclass:: RandomApply(layer, rate=0.5, batchwise=False, seed=None, **kwargs)
.. autoclass:: RandomChoice(layers, batchwise=False, seed=None, **kwargs)    
.. autoclass:: RepeatedAugment(layers, shuffle=True, seed=None, **kwargs)

Preprocessing 2D
****************

**Geometry**

----

.. autoclass:: CenterCrop(height, width, padding_value=0, bounding_box_format=None, bounding_box_min_area_ratio=None, bounding_box_max_aspect_ratio=None, seed=None, **kwargs)
.. autoclass:: PadIfNeeded(min_height=None, min_width=None, height_divisor=None, width_divisor=None, position="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: Resize(height, width, interpolation="bilinear", antialias=False, crop_to_aspect_ratio=False, pad_to_aspect_ratio=False, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)

**Intensity**

----

.. autoclass:: AutoContrast(value_range, **kwargs)
.. autoclass:: Equalize(value_range, bins=256, **kwargs)
.. autoclass:: Grayscale(output_channels=3, **kwargs)
.. autoclass:: Invert(value_range, **kwargs)
.. autoclass:: Normalize(value_range, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs)
.. autoclass:: Rescale(scale, offset=0.0, **kwargs)

**Utility**

----

.. autoclass:: Identity(**kwargs)
.. autoclass:: SanitizeBoundingBox(min_size, bounding_box_format, **kwargs)

Base 2D
****************

.. autoclass:: VectorizedBaseRandomLayer(seed=None, **kwargs)

.. -----------------------------------------------------------
..                        3D
.. -----------------------------------------------------------

Base 3D
****************

WIP
