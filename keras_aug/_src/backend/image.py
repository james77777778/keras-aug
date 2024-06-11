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
