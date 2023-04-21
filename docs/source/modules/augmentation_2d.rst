API - Augmentations - 2D
=========================

.. automodule:: keras_aug.augmentation

.. autosummary::
    :nosignatures:

    CenterCrop
    PadIfNeeded
    RandomAffine
    RandomCropAndResize
    ResizeByLongestSide
    ResizeBySmallestSide

    CLAHE
    Normalize
    RandomBlur
    RandomBrightnessContrast
    RandomColorJitter
    RandomGamma
    RandomHSV
    RandomJpegQuality
    Rescaling

    MixUp
    MosaicYOLOV8

    ChannelDropout

    RandomApply

    VectorizedBaseRandomLayer

.. -----------------------------------------------------------
..                        Geometry
.. -----------------------------------------------------------

Geometry
---------------

.. autoclass:: CenterCrop(height, width, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: PadIfNeeded(min_height=None, min_width=None, height_divisor=None, width_divisor=None, position="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomAffine(rotation_factor, translation_height_factor, translation_width_factor, zoom_height_factor, zoom_width_factor, shear_height_factor, shear_width_factor, interpolation="bilinear", fill_mode="constant", fill_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: RandomCropAndResize(height, width, crop_area_factor, aspect_ratio_factor, interpolation="bilinear", bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeAndPad(height, width, interpolation="bilinear", antialias=False, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeByLongestSide(max_size, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: ResizeBySmallestSide(min_size, interpolation="bilinear", antialias=False, bounding_box_format=None, seed=None, **kwargs)

.. -----------------------------------------------------------
..                        Intensity
.. -----------------------------------------------------------

Intensity
---------------

.. autoclass:: CLAHE
.. autoclass:: Normalize
.. autoclass:: RandomBlur
.. autoclass:: RandomBrightnessContrast
.. autoclass:: RandomColorJitter
.. autoclass:: RandomGamma
.. autoclass:: RandomHSV
.. autoclass:: RandomJpegQuality
.. autoclass:: Rescaling(scale, offset=0.0)

.. -----------------------------------------------------------
..                        Mix
.. -----------------------------------------------------------

Mix
---------------

.. autoclass:: MixUp
.. autoclass:: MosaicYOLOV8

.. -----------------------------------------------------------
..                        Regularization
.. -----------------------------------------------------------

Regularization
---------------

.. autoclass:: ChannelDropout

.. -----------------------------------------------------------
..                        Utility
.. -----------------------------------------------------------

Utility
---------------

.. autoclass:: RandomApply

.. -----------------------------------------------------------
..                        Base
.. -----------------------------------------------------------

Base
---------------

.. autoclass:: VectorizedBaseRandomLayer
