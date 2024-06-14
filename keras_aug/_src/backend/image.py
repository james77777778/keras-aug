import math

from keras import backend

from keras_aug._src.backend.dynamic_backend import DynamicBackend


class ImageBackend(DynamicBackend):
    def __init__(self, name=None):
        super().__init__(name=name)

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
        degree_to_radian_factor = 1.0 / 180.0 * math.pi
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
            ]
        )
        matrix = ops.numpy.reshape(matrix, [batch_size, 3, 3])
        return matrix

    def rgb_to_grayscale(self, images, data_format=None):
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        # Convert to floats
        images = ops.core.convert_to_tensor(images)
        original_dtype = images.dtype
        compute_dtype = backend.result_type(original_dtype, float)
        images = ops.core.cast(images, compute_dtype)
        rgb_weights = ops.core.convert_to_tensor(
            [0.2989, 0.5870, 0.1140], dtype=compute_dtype
        )
        images = ops.numpy.tensordot(
            images, rgb_weights, axes=(channels_axis, -1)
        )
        images = ops.core.cast(images, original_dtype)
        images = ops.numpy.expand_dims(images, axis=channels_axis)
        images = ops.numpy.repeat(images, 3, axis=channels_axis)
        return images

    def rgb_to_hsv(self, images, data_format=None):
        # Ref: dm-pix
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        images = ops.numpy.where(ops.numpy.abs(images) < 1e-7, 0.0, images)
        r, g, b = ops.numpy.split(images, 3, channels_axis)
        r = ops.numpy.squeeze(r, channels_axis)
        g = ops.numpy.squeeze(g, channels_axis)
        b = ops.numpy.squeeze(b, channels_axis)

        def rgb_planes_to_hsv_planes(r, g, b):
            value = ops.numpy.maximum(ops.numpy.maximum(r, g), b)
            minimum = ops.numpy.minimum(ops.numpy.minimum(r, g), b)
            range_ = value - minimum

            safe_value = ops.numpy.where(value > 0, value, 1.0)
            safe_range = ops.numpy.where(range_ > 0, range_, 1.0)

            saturation = ops.numpy.where(value > 0, range_ / safe_value, 0.0)
            norm = 1.0 / (6.0 * safe_range)

            hue = ops.numpy.where(
                value == g,
                norm * (b - r) + 2.0 / 6.0,
                norm * (r - g) + 4.0 / 6.0,
            )
            hue = ops.numpy.where(value == r, norm * (g - b), hue)
            hue = ops.numpy.where(range_ > 0, hue, 0.0) + ops.cast(
                (hue < 0.0), hue.dtype
            )
            return hue, saturation, value

        return ops.numpy.stack(
            rgb_planes_to_hsv_planes(r, g, b), axis=channels_axis
        )

    def hsv_to_rgb(self, images, data_format=None):
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        h, s, v = ops.numpy.split(images, 3, channels_axis)
        h = ops.numpy.squeeze(h, channels_axis)
        s = ops.numpy.squeeze(s, channels_axis)
        v = ops.numpy.squeeze(v, channels_axis)

        def hsv_planes_to_rgb_planes(h, s, v):
            dh = ops.numpy.mod(h, 1.0) * 6.0
            dr = ops.numpy.clip(ops.numpy.abs(dh - 3.0) - 1.0, 0.0, 1.0)
            dg = ops.numpy.clip(2.0 - ops.numpy.abs(dh - 2.0), 0.0, 1.0)
            db = ops.numpy.clip(2.0 - ops.numpy.abs(dh - 4.0), 0.0, 1.0)
            one_minus_s = 1.0 - s

            red = v * (one_minus_s + s * dr)
            green = v * (one_minus_s + s * dg)
            blue = v * (one_minus_s + s * db)
            return red, green, blue

        return ops.numpy.stack(
            hsv_planes_to_rgb_planes(h, s, v), axis=channels_axis
        )

    def blend(self, images1, images2, factor, value_range=(0.0, 1.0)):
        ops = self.backend

        images1 = ops.numpy.multiply(images1, factor)
        images2 = ops.numpy.multiply(images2, (1.0 - factor))
        images = ops.numpy.add(images1, images2)
        images = ops.numpy.clip(images, value_range[0], value_range[1])
        return images

    def adjust_brightness(self, images, factor, value_range=(0.0, 1.0)):
        ops = self.backend

        images = ops.numpy.multiply(images, factor)
        images = ops.numpy.clip(images, value_range[0], value_range[1])
        return images

    def adjust_contrast(
        self, images, factor, value_range=(0.0, 1.0), data_format=None
    ):
        data_format = data_format or backend.image_data_format()

        ops = self.backend
        grayscales = self.rgb_to_grayscale(images, data_format)
        means = ops.numpy.mean(grayscales, axis=[-3, -2, -1], keepdims=True)
        images = self.blend(images, means, factor, value_range)
        return images

    def adjust_saturation(
        self, images, factor, value_range=(0.0, 1.0), data_format=None
    ):
        data_format = data_format or backend.image_data_format()

        grayscales = self.rgb_to_grayscale(images, data_format)
        images = self.blend(images, grayscales, factor, value_range)
        return images

    def adjust_hue(
        self, images, factor, value_range=(0.0, 1.0), data_format=None
    ):
        if value_range[0] != 0.0 or value_range[1] != 1.0:
            raise ValueError("`value_range` must be `(0.0, 1.0)`")
        data_format = data_format or backend.image_data_format()
        channels_axis = -1 if data_format == "channels_last" else -3

        ops = self.backend
        images = self.rgb_to_hsv(images, data_format)
        h, s, v = ops.numpy.split(images, 3, channels_axis)
        h = ops.numpy.add(h, factor)
        h = ops.numpy.mod(h, 1.0)
        images = ops.numpy.concatenate([h, s, v], channels_axis)
        images = self.hsv_to_rgb(images, data_format)
        return images
