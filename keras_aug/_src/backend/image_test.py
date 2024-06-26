from absl.testing import parameterized
from keras import backend
from keras import ops
from keras.src import testing
from keras.src.testing.test_utils import named_product

from keras_aug._src.backend.image import ImageBackend
from keras_aug._src.utils.test_utils import get_images


class ImageBackendTest(testing.TestCase, parameterized.TestCase):
    def setUp(self):
        # Defaults to channels_last
        self.data_format = backend.image_data_format()
        backend.set_image_data_format("channels_last")
        return super().setUp()

    def tearDown(self) -> None:
        backend.set_image_data_format(self.data_format)
        return super().tearDown()

    def test_crop(self):
        image_backend = ImageBackend()

        # Test channels_last
        x = get_images("float32", "channels_last")
        y = image_backend.crop(x, top=5, left=6, height=13, width=14)
        self.assertAllClose(y, x[:, 5 : 5 + 13, 6 : 6 + 14, :])

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images("float32", "channels_first")
        y = image_backend.crop(x, top=5, left=6, height=13, width=14)
        self.assertAllClose(y, x[:, :, 5 : 5 + 13, 6 : 6 + 14])

    @parameterized.named_parameters(
        named_product(mode=["constant", "reflect", "symmetric"])
    )
    def test_pad(self, mode):
        image_backend = ImageBackend()

        # Test channels_last
        x = get_images("float32", "channels_last")
        y = image_backend.pad(x, mode, 2, 3, 4, 5)

        ref_y = ops.pad(
            x,
            [[0, 0], [2, 3], [4, 5], [0, 0]],
            mode,
            0 if mode == "constant" else None,
        )
        self.assertAllClose(y, ref_y)

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = get_images("float32", "channels_first")
        y = image_backend.pad(x, mode, 2, 3, 4, 5)

        ref_y = ops.pad(
            x,
            [[0, 0], [0, 0], [2, 3], [4, 5]],
            mode,
            0 if mode == "constant" else None,
        )
        self.assertAllClose(y, ref_y)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_adjust_brightness(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.adjust_brightness(x, 0.5)

        ref_y = TF.adjust_brightness(torch.tensor(x), 0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_adjust_contrast(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.adjust_contrast(x, 0.5, "channels_first")

        ref_y = TF.adjust_contrast(torch.tensor(x), 0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_adjust_hue(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.adjust_hue(x, 0.5, "channels_first")

        ref_y = TF.adjust_hue(torch.tensor(x), 0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_adjust_saturation(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.adjust_saturation(x, 0.5, "channels_first")

        ref_y = TF.adjust_saturation(torch.tensor(x), 0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_auto_contrast(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        if dtype == "uint8":
            atol = 1
        else:
            atol = 1e-6

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.auto_contrast(x, "channels_first")

        ref_y = TF.autocontrast(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y, atol=atol)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_blend(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x1 = get_images(dtype, "channels_first")
        x2 = get_images(dtype, "channels_first")
        y = image_backend.blend(x1, x2, 0.5)

        ref_y = TF._color._blend(torch.tensor(x1), torch.tensor(x2), 0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_equalize(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        if dtype == "float32":
            atol = 1
        elif dtype == "uint8":
            atol = 1e-6
        y = image_backend.equalize(x, data_format="channels_first")

        ref_y = TF.equalize(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y, atol=atol)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_guassian_blur(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        if dtype == "float32":
            atol = 1
        elif dtype == "uint8":
            atol = 1e-6
        y = image_backend.guassian_blur(
            x, (3, 3), (0.1, 0.1), data_format="channels_first"
        )

        ref_y = TF.gaussian_blur(torch.tensor(x), (3, 3), (0.1, 0.1))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y, atol=atol)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_rgb_to_grayscale(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.rgb_to_grayscale(x, 3, "channels_first")

        ref_y = TF.rgb_to_grayscale(torch.tensor(x), 3)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

        # Test ops.image.rgb_to_grayscale
        y = ops.image.rgb_to_grayscale(x, "channels_first")

        ref_y = TF.rgb_to_grayscale(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(
        named_product(dtype=["float32", "uint8", "int8"])
    )
    def test_invert(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.invert(x)

        ref_y = TF.invert(torch.tensor(x))
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_posterize(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.posterize(x, bits=3)

        ref_y = TF.posterize(torch.tensor(x), bits=3)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_sharpen(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        y = image_backend.sharpen(x, 0.5, "channels_first")

        ref_y = TF.adjust_sharpness(torch.tensor(x), 0.5)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)

    @parameterized.named_parameters(named_product(dtype=["float32", "uint8"]))
    def test_solarize(self, dtype):
        import torch
        import torchvision.transforms.v2.functional as TF

        image_backend = ImageBackend()
        x = get_images(dtype, "channels_first")
        if backend.is_float_dtype(dtype):
            threshold = 0.5
        elif dtype == "uint8":
            threshold = 127
        y = image_backend.solarize(x, threshold=threshold)

        ref_y = TF.solarize(torch.tensor(x), threshold=threshold)
        ref_y = ref_y.cpu().numpy()
        self.assertAllClose(y, ref_y)
        self.assertDType(y, dtype)
