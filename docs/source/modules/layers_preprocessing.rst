API - Preprocessing - 2D
=========================

.. automodule:: keras_aug.layers.preprocessing

.. -----------------------------------------------------------
..                        Geometry
.. -----------------------------------------------------------

Geometry
---------------

.. autoclass:: CenterCrop(height, width, postion="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
.. autoclass:: PadIfNeeded(min_height=None, min_width=None, height_divisor=None, width_divisor=None, position="center", padding_value=0, bounding_box_format=None, seed=None, **kwargs)
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
.. autoclass:: Equalize(value_range, bins=256, **kwargs)
.. autoclass:: Grayscale(output_channels=3, **kwargs)
.. autoclass:: Invert(value_range, **kwargs)
.. autoclass:: Normalize(value_range, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs)
.. autoclass:: Rescale(scale, offset=0.0)

.. -----------------------------------------------------------
..                        Utility
.. -----------------------------------------------------------

Utility
---------------

.. autoclass:: Identity(**kwargs)
