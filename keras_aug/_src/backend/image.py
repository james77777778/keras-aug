import math

from keras import backend

from keras_aug._src.backend.dynamic_backend import DynamicBackend


class ImageBackend(DynamicBackend):
    def __init__(self, name=None):
        super().__init__(name=name)

    def crop(self, images, top, left, height, width, data_format=None):
        data_format = backend.image_data_format()

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
        data_format = backend.image_data_format()

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
