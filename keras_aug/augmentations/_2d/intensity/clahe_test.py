from functools import partial

import numpy as np
import tensorflow as tf

from keras_aug import augmentations

"""
equalize is borrowed from:
https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/image/color_ops.py#L62-L84
"""  # noqa: E501


def _scale_channel(image, channel: int, bins: int = 256) -> tf.Tensor:
    """Scale the data in the channel to implement equalize."""
    image_dtype = image.dtype
    image = tf.cast(image[:, :, channel], tf.int32)

    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(image, [0, bins - 1], nbins=bins)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero_histo = tf.boolean_mask(histo, histo != 0)
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // (bins - 1)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    if step == 0:
        result = image
    else:
        lut_values = (tf.cumsum(histo, exclusive=True) + (step // 2)) // step
        lut_values = tf.clip_by_value(lut_values, 0, bins - 1)
        result = tf.gather(lut_values, image)

    return tf.cast(result, image_dtype)


def _equalize_image(image, bins: int = 256) -> tf.Tensor:
    """Implements Equalize function from PIL using TF ops."""
    image = tf.stack(
        [_scale_channel(image, c, bins) for c in range(image.shape[-1])], -1
    )
    return image


def equalize(image, bins: int = 256) -> tf.Tensor:
    """Equalize image(s)
    Args:
      images: A tensor of shape
          `(num_images, num_rows, num_columns, num_channels)` (NHWC), or
          `(num_rows, num_columns, num_channels)` (HWC), or
          `(num_rows, num_columns)` (HW). The rank must be statically known (the
          shape is not `TensorShape(None)`).
      bins: The number of bins in the histogram.
      name: The name of the op.
    Returns:
      Image(s) with the same type and shape as `images`, equalized.
    """
    fn = partial(_equalize_image)
    image = tf.map_fn(lambda x: fn(x, bins), image)
    return image


class CLAHETest(tf.test.TestCase):
    regular_args = {
        "value_range": (0, 255),
        "factor": (2, 4),
        "tile_grid_size": (8, 8),
        "gpu_optimized": True,
    }

    def test_preserves_output_shape(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0

        layer = augmentations.CLAHE(**self.regular_args)
        output = layer(image)

        self.assertEqual(image.shape, output.shape)
        self.assertNotAllClose(image, output)

    def test_config_with_custom_name(self):
        layer = augmentations.CLAHE(
            **self.regular_args,
            name="image_preproc",
        )
        config = layer.get_config()
        layer_1 = augmentations.CLAHE.from_config(config)
        self.assertEqual(layer_1.name, layer.name)

    def test_config(self):
        layer = augmentations.CLAHE(**self.regular_args)
        config = layer.get_config()
        self.assertEqual(config["factor"], self.regular_args["factor"])
        self.assertEqual(
            config["tile_grid_size"], self.regular_args["tile_grid_size"]
        )
        self.assertEqual(
            config["gpu_optimized"], self.regular_args["gpu_optimized"]
        )

    def test_output_dtypes(self):
        inputs = np.array(
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], dtype="float64"
        )
        layer = augmentations.CLAHE(**self.regular_args)
        self.assertAllEqual(layer(inputs).dtype, "float32")

    def test_adjustment_for_non_rgb_value_range(self):
        image_shape = (4, 8, 8, 3)
        # Value range (0, 100)
        image = tf.random.uniform(shape=image_shape) * 100.0
        args = self.regular_args.copy()
        args.update({"value_range": (0, 100)})
        layer = augmentations.CLAHE(**args)

        output = layer(image)

        self.assertNotAllClose(image, output)

    def test_clahe_output(self):
        image_shape = (4, 8, 8, 3)
        image = tf.random.uniform(shape=image_shape) * 255.0
        image = tf.cast(image, dtype=tf.uint8)
        args = self.regular_args.copy()
        args.update({"factor": (1, 1), "tile_grid_size": (1, 1)})
        layer = augmentations.CLAHE(**args)
        equalized = equalize(image)

        output = layer(image)

        self.assertAllClose(output, equalized, atol=1)
