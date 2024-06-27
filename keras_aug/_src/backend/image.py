import math

from keras import backend

from keras_aug._src.backend.dynamic_backend import DynamicBackend


class ImageBackend(DynamicBackend):
    def __init__(self, name=None):
        super().__init__(name=name)

    def transform_dtype(self, images, dtype):
        # Ref: torchvision.transforms.v2.ToDtype
        ops = self.backend
        from_dtype = backend.standardize_dtype(images.dtype)
        dtype = backend.standardize_dtype(dtype)

        if from_dtype == dtype:
            return images

        is_float_input = backend.is_float_dtype(from_dtype)
        is_float_output = backend.is_float_dtype(dtype)

        if is_float_input:
            # float to float
            if is_float_output:
                return ops.cast(images, dtype)

            # float to int
            if (from_dtype == "float32" and dtype in ("int32", "int64")) or (
                from_dtype == "float64" and dtype == "int64"
            ):
                raise ValueError(
                    f"The conversion from {from_dtype} to {dtype} cannot be "
                    "performed safely."
                )
            eps = backend.epsilon()
            max_value = float(self._max_value_of_dtype(dtype))
            return ops.cast(
                ops.numpy.multiply(images, max_value + 1.0 - eps), dtype
            )
        else:
            # int to float
            if is_float_output:
                max_value = float(self._max_value_of_dtype(from_dtype))
                return ops.numpy.divide(ops.cast(images, dtype), max_value)

            # int to int
            num_bits_input = self._num_bits_of_dtype(from_dtype)
            num_bits_output = self._num_bits_of_dtype(dtype)

            if num_bits_input > num_bits_output:
                return ops.cast(
                    images >> (num_bits_input - num_bits_output), dtype
                )
            else:
                return ops.cast(images, dtype) << (
                    num_bits_output - num_bits_input
                )

    def crop(self, images, top, left, height, width, data_format=None):
        data_format = data_format or backend.image_data_format()

        ops = self.backend
        images_shape = ops.shape(images)
        start_indices = [top, left]
        lengths = [height, width]
        if data_format == "channels_last":
            start_indices = start_indices + [0]
            lengths = lengths + [images_shape[-1]]
        else:
            start_indices = [0] + start_indices
            lengths = [images_shape[-3]] + lengths
        start_indices = [0] + start_indices
        lengths = [images_shape[0]] + lengths
        images = ops.core.slice(
            images, ops.cast(start_indices, "int32"), lengths
        )
        return images

    def pad(
        self,
        images,
        mode,
        top,
        bottom,
        left,
        right,
        constant_value=0,
        data_format=None,
    ):
        data_format = data_format or backend.image_data_format()

        ops = self.backend
        if self.name == "torch":  # Workaround for torch
            top = int(top)
            bottom = int(bottom)
            left = int(left)
            right = int(right)
        pad_width = [[top, bottom], [left, right]]
        if data_format == "channels_last":
            pad_width = pad_width + [[0, 0]]
        else:
            pad_width = [[0, 0]] + pad_width
        pad_width = [[0, 0]] + pad_width  # 4D

        images = ops.numpy.pad(
            images,
            pad_width,
            mode,
            constant_value if mode == "constant" else None,
        )
        return images

    def adjust_brightness(self, images, factor):
        ops = self.backend
        images = ops.convert_to_tensor(images)
        original_dtype = backend.standardize_dtype(images.dtype)
        is_float_inputs = backend.is_float_dtype(original_dtype)
        max_value = self._max_value_of_dtype(original_dtype)

        images = ops.numpy.multiply(images, factor)
        images = ops.numpy.clip(images, 0, max_value)
        if not is_float_inputs:
            images = ops.cast(images, original_dtype)
        return images

    def adjust_contrast(self, images, factor, data_format=None):
        data_format = data_format or backend.image_data_format()

        ops = self.backend
        images = ops.convert_to_tensor(images)
        original_dtype = backend.standardize_dtype(images.dtype)
        is_float_inputs = backend.is_float_dtype(original_dtype)

        grayscales = ops.image.rgb_to_grayscale(images, data_format)
        if not is_float_inputs:
            grayscales = ops.numpy.floor(grayscales)
        means = ops.numpy.mean(grayscales, axis=[-3, -2, -1], keepdims=True)
        images = self.blend(images, means, factor)
        return images

    def adjust_saturation(self, images, factor, data_format=None):
        data_format = data_format or backend.image_data_format()

        ops = self.backend
        images = ops.convert_to_tensor(images)
        original_dtype = backend.standardize_dtype(images.dtype)
        is_float_inputs = backend.is_float_dtype(original_dtype)

        grayscales = ops.image.rgb_to_grayscale(images, data_format)
        if not is_float_inputs:
            grayscales = ops.numpy.floor(grayscales)
        images = self.blend(images, grayscales, factor)
        return images

    def adjust_hue(self, images, factor, data_format=None):
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        images = ops.convert_to_tensor(images)
        original_dtype = backend.standardize_dtype(images.dtype)
        max_value = self._max_value_of_dtype(original_dtype)

        images = self.transform_dtype(
            images, backend.result_type(original_dtype, float)
        )
        images = ops.image.rgb_to_hsv(images, data_format)
        h, s, v = ops.numpy.split(images, 3, channels_axis)
        h = ops.numpy.add(h, factor)
        h = ops.numpy.mod(h, 1.0)
        images = ops.numpy.concatenate([h, s, v], channels_axis)
        images = ops.image.hsv_to_rgb(images, data_format)
        images = ops.numpy.clip(images, 0, max_value)
        images = self.transform_dtype(images, original_dtype)
        return images

    def auto_contrast(self, images, data_format=None):
        data_format = data_format or backend.image_data_format()
        axis = (1, 2) if data_format == "channels_last" else (2, 3)

        ops = self.backend
        original_dtype = backend.standardize_dtype(images.dtype)
        max_value = self._max_value_of_dtype(original_dtype)
        if not backend.is_float_dtype(original_dtype):
            images = ops.cast(
                images, backend.result_type(original_dtype, float)
            )

        lows = ops.numpy.min(images, axis=axis, keepdims=True)
        highs = ops.numpy.max(images, axis=axis, keepdims=True)
        eq_index = ops.numpy.equal(lows, highs)
        inverse_scale = ops.numpy.divide(
            ops.numpy.subtract(highs, lows), max_value
        )
        lows = ops.numpy.where(eq_index, 0.0, lows)
        inverse_scale = ops.numpy.where(eq_index, 1.0, inverse_scale)

        images = ops.numpy.divide(
            ops.numpy.subtract(images, lows), inverse_scale
        )
        images = ops.numpy.clip(images, 0, max_value)
        images = ops.cast(images, original_dtype)
        return images

    def blend(self, images1, images2, factor):
        ops = self.backend
        images1 = ops.convert_to_tensor(images1)
        images2 = ops.convert_to_tensor(images2)
        original_dtype = backend.standardize_dtype(images1.dtype)
        is_float_inputs = backend.is_float_dtype(original_dtype)
        max_value = self._max_value_of_dtype(original_dtype)

        images1 = ops.numpy.multiply(images1, factor)
        images2 = ops.numpy.multiply(images2, (1.0 - factor))
        images = ops.numpy.add(images1, images2)
        images = ops.numpy.clip(images, 0, max_value)
        if not is_float_inputs:
            images = ops.cast(images, original_dtype)
        return images

    def equalize(self, images, bins=256, data_format=None):
        if bins != 256:
            raise NotImplementedError("`bins` must be `256`.")
        data_format = data_format or backend.image_data_format()

        ops = self.backend
        images = ops.convert_to_tensor(images)
        original_dtype = backend.standardize_dtype(images.dtype)
        images_shape = ops.shape(images)
        images = self.transform_dtype(images, "uint8")

        def _scale_channel(image_channel):
            hist = ops.numpy.bincount(
                ops.numpy.reshape(image_channel, [-1]), minlength=bins
            )
            nonzero = ops.numpy.where(ops.numpy.not_equal(hist, 0), None, None)
            nonzero_hist = ops.numpy.reshape(
                ops.numpy.take(hist, nonzero), [-1]
            )
            step = ops.numpy.floor_divide(
                ops.numpy.sum(hist) - nonzero_hist[-1], 255
            )

            def step_is_0():
                return image_channel

            def step_not_0():
                lut = ops.numpy.floor_divide(
                    ops.numpy.add(
                        ops.numpy.cumsum(hist), ops.numpy.floor_divide(step, 2)
                    ),
                    step,
                )
                lut = ops.numpy.pad(lut[:-1], [[1, 0]])
                lut = ops.numpy.clip(lut, 0, 255)
                result = ops.numpy.take(lut, ops.cast(image_channel, "int64"))
                return ops.cast(result, "uint8")

            return ops.cond(step == 0, step_is_0, step_not_0)

        def _equalize_single_image(image):
            if data_format == "channels_last":
                return ops.numpy.stack(
                    [
                        _scale_channel(image[..., c])
                        for c in range(image.shape[-1])
                    ],
                    axis=-1,
                )
            else:
                return ops.numpy.stack(
                    [_scale_channel(image[c]) for c in range(image.shape[-3])],
                    axis=-3,
                )

        # Workaround for tf.data
        if self.name == "tensorflow":
            import tensorflow as tf

            images = tf.map_fn(_equalize_single_image, images)
        else:
            images = ops.numpy.stack(
                [
                    _equalize_single_image(x)
                    for x in ops.core.unstack(images, axis=0)
                ],
                axis=0,
            )
        images = ops.numpy.reshape(images, images_shape)
        images = self.transform_dtype(images, original_dtype)
        return images

    def guassian_blur(self, images, kernel_size, sigma, data_format=None):
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        images = ops.core.convert_to_tensor(images)
        channels = ops.shape(images)[channels_axis]
        original_dtype = backend.standardize_dtype(images.dtype)
        compute_dtype = backend.result_type(original_dtype, float)

        def _get_gaussian_kernel_1d(kernel_size, sigma, dtype):
            lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0) * sigma)
            x = ops.numpy.linspace(-lim, lim, num=kernel_size, dtype=dtype)
            kernel_1d = ops.nn.softmax(-ops.numpy.power(x, 2), axis=0)
            return kernel_1d

        def _get_gaussian_kernel_2d(kernel_size, sigma, dtype):
            kernel_1d_x = _get_gaussian_kernel_1d(
                kernel_size[0], sigma[0], dtype
            )
            kernel_1d_y = _get_gaussian_kernel_1d(
                kernel_size[1], sigma[1], dtype
            )
            kernel_2d = ops.numpy.multiply(
                ops.numpy.expand_dims(kernel_1d_y, axis=-1), kernel_1d_x
            )
            return kernel_2d

        images = ops.cast(images, compute_dtype)
        kernel = _get_gaussian_kernel_2d(kernel_size, sigma, compute_dtype)

        kernel = ops.numpy.expand_dims(kernel, axis=(2, 3))
        kernel = ops.numpy.tile(kernel, [1, 1, channels, 1])

        # Padding
        pad_width = [
            [kernel_size[1] // 2, kernel_size[1] // 2],  # y
            [kernel_size[0] // 2, kernel_size[0] // 2],  # x
        ]
        if data_format == "channels_last":
            pad_width = pad_width + [[0, 0]]
        else:
            pad_width = [[0, 0]] + pad_width
        pad_width = [[0, 0]] + pad_width
        images = ops.numpy.pad(images, pad_width, mode="reflect")
        images = ops.nn.depthwise_conv(images, kernel, data_format=data_format)
        if backend.is_int_dtype(original_dtype):
            images = ops.numpy.round(images)
            images = ops.cast(images, original_dtype)
        return images

    def rgb_to_grayscale(self, images, num_channels=3, data_format=None):
        if num_channels not in (1, 3):
            raise ValueError(
                "`num_channels` must be 1 or 3. "
                f"Received: num_channels={num_channels}"
            )
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        images = ops.core.convert_to_tensor(images)
        original_dtype = backend.standardize_dtype(images.dtype)
        images = ops.image.rgb_to_grayscale(images, data_format)
        images = ops.core.cast(images, original_dtype)
        if num_channels == 3:
            images = ops.numpy.repeat(images, 3, axis=channels_axis)
        return images

    def invert(self, images):
        ops = self.backend
        images = ops.convert_to_tensor(images)
        dtype = backend.standardize_dtype(images.dtype)
        if backend.is_float_dtype(dtype):
            return ops.numpy.subtract(1.0, images)
        elif "uint" in dtype:
            return ~images
        else:
            # signed integer dtypes
            num_bits = self._num_bits_of_dtype(dtype)
            return images ^ ((1 << num_bits) - 1)

    def posterize(self, images, bits):
        if not isinstance(bits, int):
            raise TypeError(
                "`bits` must be an integer. "
                f"Received: bits={bits} of type {type(bits)}"
            )

        ops = self.backend
        images = ops.convert_to_tensor(images)
        dtype = backend.standardize_dtype(images.dtype)

        def posterize_float(images):
            levels = 1 << bits
            images = ops.numpy.floor(ops.numpy.multiply(images, levels))
            images = ops.numpy.clip(images, 0, levels - 1)
            images = ops.numpy.multiply(images, 1.0 / levels)
            return images

        def posterize_int(images):
            dtype_bits = self._num_bits_of_dtype(dtype)
            if bits >= dtype_bits:
                return images
            mask = ((1 << bits) - 1) << (dtype_bits - bits)
            return images & mask

        if backend.is_float_dtype(dtype):
            images = posterize_float(images)
        else:
            images = posterize_int(images)
        return images

    def sharpen(self, images, factor, data_format=None):
        data_format = data_format or backend.image_data_format()
        if data_format == "channels_last":
            channels_axis = -1
            h_axis = -3
            w_axis = -2
        else:
            channels_axis = -3
            h_axis = -2
            w_axis = -1

        ops = self.backend
        images = ops.convert_to_tensor(images)
        channels = ops.shape(images)[channels_axis]
        original_dtype = images.dtype
        float_dtype = backend.result_type(original_dtype, float)
        max_value = self._max_value_of_dtype(original_dtype)
        images = ops.cast(images, float_dtype)
        # [1 1 1]
        # [1 5 1]
        # [1 1 1]
        kernel = (
            ops.convert_to_tensor(
                [[1, 1, 1], [1, 5, 1], [1, 1, 1]], float_dtype
            )
            / 13.0
        )
        kernel = ops.numpy.expand_dims(kernel, axis=[-1, -2])
        kernel = ops.numpy.repeat(kernel, channels, 2)
        blurred_degenerate = ops.nn.depthwise_conv(
            images, kernel, 1, data_format=data_format
        )
        if backend.is_int_dtype(original_dtype):
            blurred_degenerate = ops.numpy.round(blurred_degenerate)

        if data_format == "channels_last":
            view = images[:, 1:-1, 1:-1, :]
            top, bottom = images[:, 0:1, 1:-1, :], images[:, -1:, 1:-1, :]
            left, right = images[:, :, 0:1, :], images[:, :, -1:, :]
        else:
            view = images[:, :, 1:-1, 1:-1]
            top, bottom = images[:, :, 0:1, 1:-1], images[:, :, -1:, 1:-1]
            left, right = images[:, :, :, 0:1], images[:, :, :, -1:]

        # We speed up blending by minimizing flops and doing in-place.
        # The 2 blend options are mathematically equivalent:
        # x + (1-r) * (y-x) = x + (1-r) * y - (1-r) * x = x * r + y * (1-r)
        view = ops.numpy.add(
            view,
            ops.numpy.multiply(
                1.0 - factor, ops.numpy.subtract(blurred_degenerate, view)
            ),
        )

        # Fix the borders of the resulting images by filling in the values of
        # the original images
        images = ops.numpy.concatenate([top, view, bottom], axis=h_axis)
        images = ops.numpy.concatenate([left, images, right], axis=w_axis)

        images = ops.numpy.clip(images, 0, max_value)
        images = ops.cast(images, original_dtype)
        return images

    def solarize(self, images, threshold):
        ops = self.backend
        images = ops.convert_to_tensor(images)
        dtype = backend.standardize_dtype(images.dtype)
        max_value = self._max_value_of_dtype(dtype)
        if threshold > max_value or threshold < 0:
            raise ValueError(
                "`threshold` should be less or equal to the maximum value of "
                "the input dtype. "
                f"Received: threshold={threshold}, images.dtype={dtype}"
            )
        images = ops.numpy.where(
            images >= ops.cast(threshold, images.dtype),
            self.invert(images),
            images,
        )
        return images

    def compute_affine_matrix(
        self,
        center_x,
        center_y,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
    ):
        ops = self.backend
        batch_size = ops.shape(angle)[0]
        dtype = angle.dtype
        width = ops.cast(width, dtype)
        height = ops.cast(height, dtype)
        cx = center_x * width
        cy = center_y * height
        degree_to_radian_factor = math.pi / 180.0
        rot = ops.numpy.multiply(angle, degree_to_radian_factor)
        tx = -translate_x * width
        ty = -translate_y * height
        sx = ops.numpy.multiply(shear_x, degree_to_radian_factor)
        sy = ops.numpy.multiply(shear_y, degree_to_radian_factor)

        # Compute rotation & scaling & shear & translation matrix
        # cv2.getRotationMatrix2D
        alpha = ops.numpy.cos(rot) / scale
        beta = ops.numpy.sin(rot) / scale
        matrix = ops.numpy.stack(
            [
                alpha,
                # + sx (shear)
                beta + sx,
                # - cx * sx (shear) + tx (translate)
                (1.0 - alpha) * cx - beta * cy - cx * sx + tx,
                # + sy (shear)
                -beta + sy,
                alpha,
                # - cy * sy (shear) + ty (translate)
                beta * cx + (1.0 - alpha) * cy - cy * sy + ty,
                ops.numpy.zeros([batch_size], dtype),
                ops.numpy.zeros([batch_size], dtype),
                ops.numpy.ones([batch_size], dtype),
            ],
            axis=-1,
        )
        matrix = ops.numpy.reshape(matrix, [batch_size, 3, 3])
        return matrix

    def compute_inverse_affine_matrix(
        self,
        center_x,
        center_y,
        angle,
        translate_x,
        translate_y,
        scale,
        shear_x,
        shear_y,
        height,
        width,
    ):
        # Ref: TF._geometry._get_inverse_affine_matrix
        ops = self.backend
        batch_size = ops.shape(angle)[0]
        dtype = angle.dtype
        width = ops.cast(width, dtype)
        height = ops.cast(height, dtype)

        angle = -angle
        shear_x = -shear_x
        shear_y = -shear_y

        cx = center_x * width
        cy = center_y * height
        rot = ops.numpy.multiply(angle, 1.0 / 180.0 * math.pi)
        tx = -translate_x * width
        ty = -translate_y * height
        sx = ops.numpy.multiply(shear_x, 1.0 / 180.0 * math.pi)
        sy = ops.numpy.multiply(shear_y, 1.0 / 180.0 * math.pi)

        # Cached results
        cos_sy = ops.numpy.cos(sy)
        tan_sx = ops.numpy.tan(sx)
        rot_minus_sy = rot - sy
        cx_plus_tx = cx + tx
        cy_plus_ty = cy + ty

        # Rotate Scale Shear (RSS) without scaling
        a = ops.numpy.cos(rot_minus_sy) / cos_sy
        b = -(a * tan_sx + ops.numpy.sin(rot))
        c = ops.numpy.sin(rot_minus_sy) / cos_sy
        d = ops.numpy.cos(rot) - c * tan_sx

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        a0 = d * scale
        a1 = -b * scale
        b0 = -c * scale
        b1 = a * scale
        a2 = cx - a0 * cx_plus_tx - a1 * cy_plus_ty
        b2 = cy - b0 * cx_plus_tx - b1 * cy_plus_ty

        # Shape of matrix: [[batch_size], ...] -> [batch_size, 6]
        matrix = ops.numpy.stack(
            [
                a0,
                a1,
                a2,
                b0,
                b1,
                b2,
                ops.numpy.zeros([batch_size], dtype),
                ops.numpy.zeros([batch_size], dtype),
                ops.numpy.ones([batch_size], dtype),
            ],
            axis=-1,
        )
        matrix = ops.numpy.reshape(matrix, [batch_size, 3, 3])
        return matrix

    def fill_rectangles(self, images, fill_images, boxes, data_format=None):
        data_format = data_format or backend.image_data_format()
        if data_format == "channels_last":
            h_axis, w_axis, c_axis = -3, -2, -1
        else:
            c_axis, h_axis, w_axis = -3, -2, -1

        ops = self.backend
        images_shape = ops.shape(images)
        height, width = images_shape[h_axis], images_shape[w_axis]
        x0, y0, x1, y1 = boxes

        def _to_mask(start, end, mask_len):
            batch_size = ops.shape(start)[0]
            axis_indices = ops.numpy.arange(mask_len, dtype=start.dtype)
            axis_indices = ops.numpy.expand_dims(axis_indices, 0)
            axis_indices = ops.numpy.tile(axis_indices, [batch_size, 1])

            start = ops.numpy.expand_dims(start, axis=-1)
            end = ops.numpy.expand_dims(end, axis=-1)
            mask = ops.numpy.logical_and(
                ops.numpy.greater_equal(axis_indices, start),
                ops.numpy.less(axis_indices, end),
            )
            return mask

        w_masks = _to_mask(x0, x1, width)
        h_masks = _to_mask(y0, y1, height)
        w_masks = ops.numpy.expand_dims(w_masks, axis=-2)
        h_masks = ops.numpy.expand_dims(h_masks, axis=-1)
        masks = ops.numpy.logical_and(w_masks, h_masks)
        masks = ops.numpy.expand_dims(masks, axis=c_axis)
        return ops.numpy.where(masks, fill_images, images)

    def _max_value_of_dtype(self, dtype):
        dtype = backend.standardize_dtype(dtype)
        if dtype == "uint8":
            return 255
        elif dtype == "uint16":
            return 65535
        elif dtype == "uint32":
            return 4294967295
        elif dtype == "uint64":
            return 18446744073709551615
        elif dtype == "int8":
            return 127
        elif dtype == "int16":
            return 32767
        elif dtype == "int32":
            return 2147483647
        elif dtype == "int64":
            return 9223372036854775807
        else:
            return 1

    def _num_bits_of_dtype(self, dtype):
        dtype = backend.standardize_dtype(dtype)
        if dtype == "uint8":
            return 8
        elif dtype == "uint16":
            return 16
        elif dtype == "uint32":
            return 32
        elif dtype == "uint64":
            return 64
        elif dtype == "int8":
            return 7
        elif dtype == "int16":
            return 15
        elif dtype == "int32":
            return 31
        elif dtype == "int64":
            return 63
        else:
            raise ValueError(
                "_num_bits_of_dtype is only defined for integer dtypes. "
                f"Received: dtype={dtype}"
            )
