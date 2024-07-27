import os

os.environ["KERAS_BACKEND"] = "torch"  # torch is fastest for eager mode

import gradio as gr
import keras
from keras import ops

from keras_aug import layers as ka_layers
from keras_aug import ops as ka_ops

# Images
astronaut = keras.utils.load_img(
    keras.utils.get_file(
        "astronaut.jpg",
        "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/astronaut.jpg",
    ),
    target_size=(512, 512),
)
dog1 = keras.utils.load_img(
    keras.utils.get_file(
        "dog1.jpg",
        "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog1.jpg",
    ),
    target_size=(512, 512),
)
dog2 = keras.utils.load_img(
    keras.utils.get_file(
        "dog2.jpg",
        "https://raw.githubusercontent.com/pytorch/vision/main/gallery/assets/dog2.jpg",
    ),
    target_size=(512, 512),
)


def center_crop(image, height, width):
    output = ka_layers.vision.CenterCrop((height, width))(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def color_jitter(image, brightness, contrast, saturation, hue):
    output = ka_layers.vision.ColorJitter(
        brightness, contrast, saturation, hue
    )(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def cut_mix(image1, image2, alpha):
    h, w = image1.shape[0], image1.shape[1]
    image2 = ka_layers.vision.Resize([h, w], dtype="uint8")(image2)
    input = ops.stack([image1, image2], axis=0)
    output = ka_layers.vision.CutMix(alpha)(input)[0]
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def gaussian_blur(image, kernel_size, sigma):
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    output = ka_layers.vision.GaussianBlur(kernel_size, sigma)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def gaussian_noise(image, mean, sigma):
    output = ka_layers.vision.GaussianNoise(mean, sigma)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def mix_up(image1, image2, alpha):
    h, w = image1.shape[0], image1.shape[1]
    image2 = ka_layers.vision.Resize([h, w], dtype="uint8")(image2)
    input = ops.stack([image1, image2], axis=0)
    output = ka_layers.vision.MixUp(alpha)(input)[0]
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def mosaic(image1, image2, image3, image4, offset0, offset1, padding_value):
    h, w = image1.shape[0], image1.shape[1]
    image2 = ka_layers.vision.Resize([h, w], dtype="uint8")(image2)
    image3 = ka_layers.vision.Resize([h, w], dtype="uint8")(image3)
    image4 = ka_layers.vision.Resize([h, w], dtype="uint8")(image4)
    output = ka_layers.vision.Mosaic(
        (2 * h, 2 * w), (offset0, offset1), padding_value=padding_value
    )(image1, image2, image3, image4)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def pad(image, height, width, padding_value):
    output = ka_layers.vision.Pad((height, width), padding_value=padding_value)(
        image
    )
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def rand_augment(image, num_ops, magnitude):
    output = ka_layers.vision.RandAugment(num_ops, magnitude)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_affine(image, degree, translate0, translate1, scale, shear0, shear1):
    output = ka_layers.vision.RandomAffine(
        degree, (translate0, translate1), scale, (shear0, shear1)
    )(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_auto_contrast(image, p):
    output = ka_layers.vision.RandomAutoContrast(p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_channel_permutation(image):
    output = ka_layers.vision.RandomChannelPermutation(num_channels=3)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_crop(image, height, width):
    output = ka_layers.vision.RandomCrop((height, width))(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_equalize(image, p):
    output = ka_layers.vision.RandomEqualize(p=p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_erasing(image, p, scale0, scale1, ratio0, ratio1):
    output = ka_layers.vision.RandomErasing(
        p,
        (scale0, scale1),
        (ratio0, ratio1),
    )(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_flip(image, mode, p):
    output = ka_layers.vision.RandomFlip(mode, p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_grayscale(image, p):
    output = ka_layers.vision.RandomGrayscale(p=p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_hsv(image, hue, saturation, value):
    output = ka_layers.vision.RandomHSV(hue, saturation, value)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_invert(image, p):
    output = ka_layers.vision.RandomInvert(p=p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_posterize(image, bits, p):
    output = ka_layers.vision.RandomPosterize(bits, p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_resized_crop(image, size0, size1, scale0, scale1, ratio0, ratio1):
    output = ka_layers.vision.RandomResizedCrop(
        (size0, size1), (scale0, scale1), (ratio0, ratio1)
    )(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_rotation(image, degree):
    output = ka_layers.vision.RandomRotation(degree)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_sharpen(image, sharpness_factor, p):
    output = ka_layers.vision.RandomSharpen(sharpness_factor, p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def random_solarize(image, threshold, p):
    threshold = ops.convert_to_tensor(threshold, "uint8")
    threshold = ka_ops.image.transform_dtype(threshold, "uint8", "float32")
    output = ka_layers.vision.RandomSolarize(threshold, p)(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def resize(image, height, width):
    output = ka_layers.vision.Resize((height, width))(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


def trivial_augment_wide(image):
    output = ka_layers.vision.TrivialAugmentWide()(image)
    return (ops.convert_to_numpy(output) * 255.0).astype("uint8")


with gr.Blocks() as app:
    gr.Markdown(
        """
        # Demo App for KerasAug
        """
    )
    with gr.Tab("CenterCrop"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(256, label="size[0]"),
                    gr.Number(256, label="size[1]"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(center_crop, inputs=[image, *args], outputs=outputs)

    with gr.Tab("ColorJitter"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(0.5, label="brightness", minimum=0.0, step=0.1),
                    gr.Number(0.5, label="contrast", minimum=0.0, step=0.1),
                    gr.Number(0.5, label="saturation", minimum=0.0, step=0.1),
                    gr.Number(
                        0.1, label="hue", minimum=-0.5, maximum=0.5, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(color_jitter, inputs=[image, *args], outputs=outputs)

    with gr.Tab("CutMix"):
        with gr.Row():
            with gr.Column(scale=2):
                image1 = gr.Image(dog1, label="Image 1")
                image2 = gr.Image(dog2, label="Image 2")
            with gr.Column():
                args = [
                    gr.Number(1.0, label="alpha"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(cut_mix, inputs=[image1, image2, *args], outputs=outputs)

    with gr.Tab("GaussianBlur"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(5, label="kernel_size"),
                    gr.Number(1.0, label="sigma", step=0.1, maximum=1.9),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(gaussian_blur, inputs=[image, *args], outputs=outputs)

    with gr.Tab("GaussianNoise"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(0.0, label="mean", step=0.1),
                    gr.Number(0.1, label="sigma", step=0.1),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(gaussian_noise, inputs=[image, *args], outputs=outputs)

    with gr.Tab("MixUp"):
        with gr.Row():
            with gr.Column(scale=2):
                image1 = gr.Image(dog1, label="Image 1")
                image2 = gr.Image(dog2, label="Image 2")
            with gr.Column():
                args = [
                    gr.Number(1.0, label="alpha"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(mix_up, inputs=[image1, image2, *args], outputs=outputs)

    with gr.Tab("Mosaic"):
        with gr.Row():
            with gr.Column(scale=2):
                image1 = gr.Image(astronaut, label="Image 1")
                image2 = gr.Image(dog1, label="Image 2")
                image3 = gr.Image(dog2, label="Image 3")
                image4 = gr.Image(astronaut, label="Image 4")
            with gr.Column():
                args = [
                    gr.Number(0.25, label="offset[0]", step=0.01),
                    gr.Number(0.75, label="offset[1]", step=0.01),
                    gr.Number(114, label="padding_value"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(
            mosaic,
            inputs=[image1, image2, image3, image4, *args],
            outputs=outputs,
        )

    with gr.Tab("Pad"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(600, label="size[0]"),
                    gr.Number(600, label="size[1]"),
                    gr.Number(114, label="padding_value"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(pad, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandAugment"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(2, label="num_ops"),
                    gr.Number(9, label="magnitude"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(rand_augment, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomAffine"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(30, label="degree"),
                    gr.Number(0.2, label="translate[0]", step=0.1),
                    gr.Number(0.2, label="translate[1]", step=0.1),
                    gr.Number(0.5, label="scale", step=0.1),
                    gr.Number(0.2, label="shear[0]", step=0.1),
                    gr.Number(0.0, label="shear[1]", step=0.1),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_affine, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomAutoContrast"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(
            random_auto_contrast, inputs=[image, *args], outputs=outputs
        )

    with gr.Tab("RandomChannelPermutation"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = []
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(
            random_channel_permutation, inputs=[image, *args], outputs=outputs
        )

    with gr.Tab("RandomCrop"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(256, label="size[0]"),
                    gr.Number(256, label="size[1]"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_crop, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomEqualize"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_equalize, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomErasing"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                    gr.Number(0.02, label="scale[0]", step=0.01),
                    gr.Number(0.33, label="scale[1]", step=0.01),
                    gr.Number(0.3, label="ratio[0]", step=0.1),
                    gr.Number(3.3, label="ratio[1]", step=0.1),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_erasing, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomFlip"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Radio(
                        ["horizontal", "vertical", "horizontal_and_vertical"],
                        value="horizontal",
                        label="mode",
                    ),
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_flip, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomGrayscale"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_grayscale, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomHSV"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(0.015, label="hue", minimum=0.0, step=0.001),
                    gr.Number(0.7, label="saturation", minimum=0.0, step=0.1),
                    gr.Number(0.4, label="value", minimum=0.0, step=0.1),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_hsv, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomInvert"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_invert, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomPosterize"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(4, label="bits", minimum=0.0, maximum=8),
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_posterize, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomResizedCrop"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(256, label="size[0]"),
                    gr.Number(256, label="size[1]"),
                    gr.Number(0.08, label="scale[0]", step=0.01),
                    gr.Number(1.0, label="scale[1]", step=0.01),
                    gr.Number(3 / 4, label="ratio[0]", step=0.1),
                    gr.Number(4 / 3, label="ratio[1]", step=0.1),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(
            random_resized_crop, inputs=[image, *args], outputs=outputs
        )

    with gr.Tab("RandomRotation"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [gr.Number(30, label="degree")]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_rotation, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomSharpen"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(2.0, label="sharpness_factor", step=0.1),
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_sharpen, inputs=[image, *args], outputs=outputs)

    with gr.Tab("RandomSolarize"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(127, label="threshold", minimum=0, maximum=255),
                    gr.Number(
                        1.0, label="p", minimum=0.0, maximum=1.0, step=0.1
                    ),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(random_solarize, inputs=[image, *args], outputs=outputs)

    with gr.Tab("Resize"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = [
                    gr.Number(256, label="size[0]"),
                    gr.Number(256, label="size[1]"),
                ]
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(resize, inputs=[image, *args], outputs=outputs)

    with gr.Tab("TrivialAugmentWide"):
        with gr.Row():
            with gr.Column(scale=2):
                image = gr.Image(astronaut, label="Image")
            with gr.Column():
                args = []
                button = gr.Button("Run")
            with gr.Column(scale=2):
                outputs = gr.Image(label="Output")
        button.click(
            trivial_augment_wide, inputs=[image, *args], outputs=outputs
        )

if __name__ == "__main__":
    app.launch()
