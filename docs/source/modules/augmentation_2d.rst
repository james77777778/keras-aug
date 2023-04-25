API - Augmentation - 2D
=========================

.. automodule:: keras_aug.augmentation

.. -----------------------------------------------------------
..                        Geometry
.. -----------------------------------------------------------

Geometry
---------------

.. autoclass:: CenterCrop(height, width, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: PadIfNeeded(min_height=None, min_width=None, height_divisor=None, width_divisor=None, position="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomAffine(rotation_factor, translation_height_factor, translation_width_factor, zoom_height_factor, zoom_width_factor, shear_height_factor, shear_width_factor, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomCrop(height, width, interpolation="bilinear", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomCropAndResize(height, width, crop_area_factor, aspect_ratio_factor, interpolation="bilinear", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomFlip(mode="horizontal", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomRotate(factor, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: Resize(height, width, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeAndCrop(height, width, interpolation="bilinear", antialias=False, postion="center", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeAndPad(height, width, interpolation="bilinear", antialias=False, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeByLongestSide(max_size, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeBySmallestSide(min_size, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)

.. -----------------------------------------------------------
..                        Intensity
.. -----------------------------------------------------------

Intensity
---------------

.. autoclass:: AutoContrast(value_range, **kwargs)
.. autoclass:: ChannelShuffle(groups=3, seed=None, **kwargs)
.. autoclass:: Equalize(value_range, bins=256, **kwargs)
.. autoclass:: Grayscale(output_channels=3, **kwargs)
.. autoclass:: Normalize(value_range, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs)
.. autoclass:: RandomBlur(factor, seed=None, **kwargs)
.. autoclass:: RandomBrightnessContrast(value_range, brightness_factor, contrast_factor, seed=None, **kwargs)
.. autoclass:: RandomChannelShift(value_range, factor, channels=3, seed=None, **kwargs)
.. autoclass:: RandomCLAHE(value_range, factor=(1, 4), tile_grid_size=(8, 8), seed=None, **kwargs)
.. autoclass:: RandomColorJitter(value_range, brightness_factor, contrast_factor, saturation_factor, hue_factor, seed=None, **kwargs)
.. autoclass:: RandomGamma(value_range, factor, seed=None, **kwargs)
.. autoclass:: RandomHSV(value_range, hue_factor, saturation_factor, value_factor, seed=None, **kwargs)
.. autoclass:: RandomJpegQuality(value_range, factor, seed=None, **kwargs)
.. autoclass:: Rescaling(scale, offset=0.0)

.. -----------------------------------------------------------
..                        Mix
.. -----------------------------------------------------------

Mix
---------------

.. autoclass:: MixUp(alpha=0.2, seed=None, **kwargs)
.. autoclass:: MosaicYOLOV8(height, width, offset=(0.25, 0.75), seed=None, **kwargs)

.. -----------------------------------------------------------
..                        Regularization
.. -----------------------------------------------------------

Regularization
---------------

.. autoclass:: ChannelDropout(factor=(0, 2), fill_value=0, seed=None, **kwargs)

.. -----------------------------------------------------------
..                        Utility
.. -----------------------------------------------------------

Utility
---------------

.. autoclass:: RandomApply(layer, rate=0.5, seed=None, **kwargs)

.. -----------------------------------------------------------
..                        Base
.. -----------------------------------------------------------

Base
---------------

.. autoclass:: VectorizedBaseRandomLayer(seed=None, **kwargs)
