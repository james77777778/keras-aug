import math
import time

import keras_cv
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torchvision.transforms.v2 as transforms
from keras_cv import bounding_box
from torchvision import datapoints
from tqdm import tqdm

import keras_aug


def load_voc_dataset(
    name="voc/2007",
    height=640,
    width=640,
    batch_size=128,
    bounding_box_format="xyxy",
):
    def preprocess_voc(inputs, bounding_box_format):
        image = inputs["image"]
        image = tf.stop_gradient(tf.cast(image, tf.float32))
        labels = tf.random.uniform(shape=(1,), minval=0, maxval=10)
        boxes = bounding_box.convert_format(
            tf.stop_gradient(inputs["objects"]["bbox"]),
            images=image,
            source="rel_yxyx",
            target=bounding_box_format,
            dtype=tf.float32,
        )
        classes = tf.stop_gradient(
            tf.cast(inputs["objects"]["label"], tf.float32)
        )
        bounding_boxes = {"boxes": boxes, "classes": classes}
        return {
            "images": image,
            "labels": labels,
            "bounding_boxes": bounding_boxes,
        }

    resize = keras_aug.layers.ResizeAndPad(
        height=height,
        width=width,
        bounding_box_format=bounding_box_format,
        dtype=tf.uint8,
    )
    dataset = tfds.load(name, split=tfds.Split.TRAIN)
    dataset = dataset.repeat(10)
    dataset = dataset.map(
        lambda x: preprocess_voc(x, bounding_box_format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(
            batch_size, drop_remainder=True
        )
    )
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(1)
    return dataset


def tf2torch(tf_batch_data, batch_size=128):
    results = []
    for i in range(batch_size):
        torch_images = tf_batch_data["images"][i].numpy()
        torch_images = np.transpose(torch_images, axes=(2, 0, 1))
        torch_images = np.ascontiguousarray(torch_images)
        torch_images = torch.from_numpy(torch_images)
        torch_images = torch_images / 255.0  # [0.0, 255.0] -> [0.0, 1.0]

        # no labels

        tf_bounding_boxes = {
            "boxes": tf_batch_data["bounding_boxes"]["boxes"][i],
            "classes": tf_batch_data["bounding_boxes"]["classes"][i],
        }
        torch_bounding_boxes = bounding_box.to_dense(tf_bounding_boxes)
        torch_boxes = np.ascontiguousarray(
            torch_bounding_boxes["boxes"].numpy()
        )
        torch_boxes = torch.from_numpy(torch_boxes)
        torch_boxes = datapoints.BoundingBox(
            torch_boxes,
            format=datapoints.BoundingBoxFormat.XYXY,
            spatial_size=torch_images.shape[-2:],
        )

        results.append((torch_images, torch_boxes))
    return results


def compute_stats(time_costs, batch_size=128, decimals=2):
    time_costs = sorted(time_costs, reverse=True)
    # remove bottom 20%
    len_20percent = len(time_costs) // 5
    time_costs = time_costs[len_20percent:]
    # time -> FPS
    fps = [1.0 / t * batch_size for t in time_costs]
    mean = np.around(np.mean(fps), decimals=decimals)
    median = np.around(np.median(fps), decimals=decimals)
    std = np.around(np.std(fps), decimals=decimals)
    return (mean, median, std)


"""
Define the behavior to process tensor
"""


def drop_data_by_layer_name(tf_data, torch_data, layer_name):
    if layer_name == "CutMix":
        tf_data.pop("bounding_boxes")
    if layer_name == "Resize":
        tf_data.pop("labels")
    return tf_data, torch_data


def benchmark(
    benchmark_type,
    layer,
    layer_name,
    height,
    width,
    batch_size,
    bounding_box_format,
    warmup_iterations,
    benchmark_iterations,
):
    lower_benchmark_type = benchmark_type.lower()
    if lower_benchmark_type not in ("kerascv", "kerasaug", "torchvision"):
        raise NotImplementedError(f"{benchmark_type}")
    dataset = load_voc_dataset(
        height=height,
        width=width,
        batch_size=batch_size,
        bounding_box_format=bounding_box_format,
    )
    dataset_iter = iter(dataset)

    if lower_benchmark_type in ("kerascv", "kerasaug"):

        @tf.function
        def call(data):
            return layer(data)

    elif lower_benchmark_type == "torchvision":

        def call(data):
            # must call by iterating batch_inputs because torchvision applies
            # the same transformation to all the inputs of a given batch.
            # https://pytorch.org/vision/main/transforms.html
            for inputs in data:
                # inputs = (torch_images, torch_boxes)
                if isinstance(inputs, tuple):
                    _ = layer(inputs[0], inputs[1])
                else:
                    _ = layer(inputs)

    with tf.device("/cpu:0"):
        # warmup
        for _ in range(warmup_iterations):
            tf_data = next(dataset_iter)
            torch_data = tf2torch(tf_data)
            tf_data, torch_data = drop_data_by_layer_name(
                tf_data, torch_data, layer_name
            )
            if lower_benchmark_type in ("kerascv", "kerasaug"):
                _ = call(tf_data)
            elif lower_benchmark_type == "torchvision":
                _ = call(torch_data)

        # benchmark
        time_costs = []
        for _ in tqdm(
            range(benchmark_iterations),
            desc=f"{benchmark_type} {layer_name}",
            dynamic_ncols=True,
        ):
            tf_data = next(dataset_iter)
            torch_data = tf2torch(tf_data)
            tf_data, torch_data = drop_data_by_layer_name(
                tf_data, torch_data, layer_name
            )
            time.sleep(0.1)
            st = time.perf_counter()
            if lower_benchmark_type in ("kerascv", "kerasaug"):
                _ = call(tf_data)
            elif lower_benchmark_type == "torchvision":
                _ = call(torch_data)
            ed = time.perf_counter()
            time_costs.append(ed - st)
    del dataset_iter
    del dataset
    stats = compute_stats(time_costs, batch_size)
    print(f"FPS: mean={stats[0]}, median={stats[1]}, std={stats[2]}")
    return stats


if __name__ == "__main__":
    # init args
    warmup_iterations = 5
    benchmark_iterations = 20
    height = 640
    width = 640
    batch_size = 128
    bounding_box_format = "xyxy"
    skip_number = (0, 20)  # max: 20

    # candidates
    keras_aug_layers = [
        keras_aug.layers.RandomFlip(
            mode="horizontal",
            bounding_box_format=bounding_box_format,
            name="RandomHorizontalFlip",
        ),
        keras_aug.layers.RandomFlip(
            mode="vertical",
            bounding_box_format=bounding_box_format,
            name="RandomVerticalFlip",
        ),
        keras_aug.layers.RandomAffine(
            rotation_factor=(-10, 10),
            bounding_box_format=bounding_box_format,
            name="RandomRotate",
        ),
        keras_aug.layers.RandomAffine(
            rotation_factor=(-10, 10),
            translation_height_factor=0.1,
            translation_width_factor=0.1,
            zoom_height_factor=0.1,
            zoom_width_factor=0.1,
            shear_height_factor=0.1,
            shear_width_factor=0.1,
            bounding_box_format=bounding_box_format,
            name="RandomAffine",
        ),
        keras_aug.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(0.0, 2.0),
            name="RandomBrightness",
        ),
        keras_aug.layers.RandomColorJitter(
            value_range=(0, 255),
            contrast_factor=(0.5, 1.5),
            name="RandomContrast",
        ),
        keras_aug.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(0.0, 2.0),
            contrast_factor=(0.0, 2.0),
            name="RandomBrightnessContrast",
        ),
        keras_aug.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(0.0, 2.0),
            contrast_factor=(0.0, 2.0),
            saturation_factor=(0.0, 2.0),
            hue_factor=(-0.5, 0.5),
            name="RandomColorJitter",
        ),
        keras_aug.layers.Grayscale(output_channels=3, name="Grayscale"),
        keras_aug.layers.Resize(
            height=224,
            width=224,
            bounding_box_format=bounding_box_format,
            name="Resize",
        ),
        keras_aug.layers.RandomCropAndResize(
            height=224,
            width=224,
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(3 / 4, 4 / 3),
            bounding_box_format=bounding_box_format,
            name="RandomCropAndResize",
        ),
        keras_aug.layers.Equalize(value_range=(0, 255), name="Equalize"),
        keras_aug.layers.RandomGaussianBlur(
            kernel_size=11, factor=(0.1, 1.0), name="RandomGaussianBlur"
        ),
        keras_aug.layers.CutMix(name="CutMix"),
        keras_aug.layers.MixUp(alpha=32.0, name="MixUp"),
        keras_aug.layers.AutoContrast(
            value_range=(0, 255), name="AutoContrast"
        ),
        keras_aug.layers.RandomPosterize(
            value_range=(0, 255), factor=(4, 4), name="Posterize"
        ),
        keras_aug.layers.RandomSolarize(
            value_range=(0, 255), threshold_factor=(200, 200), name="Solarize"
        ),
        keras_aug.layers.RandomSharpness(
            value_range=(0, 255), factor=(1.5, 1.5), name="RandomSharpness"
        ),
        keras_aug.layers.Invert(value_range=(0, 255), name="Invert"),
    ]
    keras_cv_layers = [
        keras_cv.layers.RandomFlip(
            mode="horizontal",
            bounding_box_format=bounding_box_format,
            name="RandomHorizontalFlip",
        ),
        keras_cv.layers.RandomFlip(
            mode="vertical",
            bounding_box_format=bounding_box_format,
            name="RandomVerticalFlip",
        ),
        keras_cv.layers.RandomRotation(
            factor=(-10 / (2 * math.pi), 10 / (2 * math.pi)),
            bounding_box_format=bounding_box_format,
            name="RandomRotate",
        ),
        # use keras.Sequential to mimic RandomAffine
        tf.keras.Sequential(
            [
                # no RandomZoom due to lack of bounding_boxes support
                keras_cv.layers.RandomRotation(
                    factor=(-10 / (2 * math.pi), 10 / (2 * math.pi)),
                    bounding_box_format=bounding_box_format,
                ),
                keras_cv.layers.RandomShear(
                    x_factor=0.1,
                    y_factor=0.1,
                    bounding_box_format=bounding_box_format,
                ),
                keras_cv.layers.RandomTranslation(
                    height_factor=0.1,
                    width_factor=0.1,
                    bounding_box_format=bounding_box_format,
                ),
            ],
            name="RandomAffine",
        ),
        keras_cv.layers.RandomBrightness(
            value_range=(0, 255), factor=(-0.5, 0.5), name="RandomBrightness"
        ),
        keras_cv.layers.RandomContrast(
            value_range=(0, 255), factor=(0.0, 1.0), name="RandomContrast"
        ),
        keras_cv.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(0.0, 1.0),
            contrast_factor=(0.0, 1.0),
            saturation_factor=(0.5, 0.5),
            hue_factor=(0.0, 0.0),
            name="RandomBrightnessContrast",
        ),
        keras_cv.layers.RandomColorJitter(
            value_range=(0, 255),
            brightness_factor=(0.0, 1.0),
            contrast_factor=(0.0, 1.0),
            saturation_factor=(0.0, 1.0),
            hue_factor=(0.0, 1.0),
            name="RandomColorJitter",
        ),
        keras_cv.layers.Grayscale(output_channels=3, name="Grayscale"),
        keras_cv.layers.Resizing(
            height=224,
            width=224,
            pad_to_aspect_ratio=True,
            bounding_box_format=bounding_box_format,
            name="Resize",
        ),
        keras_cv.layers.RandomCropAndResize(
            (224, 224),
            crop_area_factor=(0.8, 1.0),
            aspect_ratio_factor=(3 / 4, 4 / 3),
            bounding_box_format=bounding_box_format,
            name="RandomCropAndResize",
        ),
        keras_cv.layers.Equalization(value_range=(0, 255), name="Equalize"),
        keras_cv.layers.RandomGaussianBlur(
            kernel_size=11, factor=(0.1, 1.0), name="RandomGaussianBlur"
        ),
        keras_cv.layers.CutMix(name="CutMix"),
        keras_cv.layers.MixUp(alpha=32.0, name="MixUp"),
        keras_cv.layers.AutoContrast(value_range=(0, 255), name="AutoContrast"),
        keras_cv.layers.Posterization(
            value_range=(0, 255), bits=4, name="Posterize"
        ),
        keras_cv.layers.Solarization(
            value_range=(0, 255), threshold_factor=(200, 200), name="Solarize"
        ),
        keras_cv.layers.RandomSharpness(
            value_range=(0, 255), factor=(0.6, 0.6), name="RandomSharpness"
        ),
        None,  # no Invert
    ]
    torchvision_layers = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=(-10, 10),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.RandomAffine(
            degrees=(-10, 10),
            translate=(0.1, 0.1),
            scale=(0.1, 0.1),
            shear=(0.1, 0.1),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.ColorJitter(brightness=(0.0, 2.0)),
        transforms.ColorJitter(contrast=(0.0, 2.0)),
        transforms.ColorJitter(brightness=(0.0, 2.0), contrast=(0.0, 2.0)),
        transforms.ColorJitter(
            brightness=(0.0, 2.0),
            contrast=(0.0, 2.0),
            saturation=(0.0, 2.0),
            hue=(-0.5, 0.5),
        ),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=False,
        ),
        transforms.RandomResizedCrop(
            (224, 224), scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3), antialias=False
        ),
        transforms.RandomEqualize(p=1.0),
        transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 1.0)),
        None,  # no CutMix
        None,  # no MixUp
        transforms.RandomAutocontrast(p=1.0),
        transforms.RandomPosterize(bits=4, p=1.0),
        None,  # cannot run RandomSolarize on float32
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=1.0),
        transforms.RandomInvert(p=1.0),
    ]

    # results
    results = {}

    # KerasAug
    name = "KerasAug"
    results[name] = {}
    for i, layer in enumerate(keras_aug_layers):
        if not (skip_number[0] <= i < skip_number[1]) or layer is None:
            continue
        layer_name = layer.name
        benchmark_stats = benchmark(
            name,
            layer,
            layer_name,
            height,
            width,
            batch_size,
            bounding_box_format,
            warmup_iterations,
            benchmark_iterations,
        )
        results[name][layer_name] = benchmark_stats

    # KerasCV
    name = "KerasCV"
    results[name] = {}
    for i, layer in enumerate(keras_cv_layers):
        if not (skip_number[0] <= i < skip_number[1]) or layer is None:
            continue
        layer_name = layer.name
        benchmark_stats = benchmark(
            name,
            layer,
            layer_name,
            height,
            width,
            batch_size,
            bounding_box_format,
            warmup_iterations,
            benchmark_iterations,
        )
        results[name][layer_name] = benchmark_stats

    # torchvision
    name = "torchvision"
    results[name] = {}
    for i, layer in enumerate(torchvision_layers):
        if not (skip_number[0] <= i < skip_number[1]) or layer is None:
            continue
        layer_name = layer.__class__.__name__
        if layer_name in results[name]:
            layer_name = f"{layer_name}_{i}"
        benchmark_stats = benchmark(
            name,
            layer,
            layer_name,
            height,
            width,
            batch_size,
            bounding_box_format,
            warmup_iterations,
            benchmark_iterations,
        )
        results[name][layer_name] = benchmark_stats

    # print results
    for package in results.keys():
        print(package)
        for op, stats in results[package].items():
            print(f"    {op}: {stats}")