import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from keras_aug.layers import RandomAffine
from keras_aug.layers import Resize
from keras_aug.utils import demo as demo_utils


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    resize = Resize(
        224, 224, crop_to_aspect_ratio=True, bounding_box_format="xyxy"
    )
    layer = RandomAffine(
        rotation_factor=(-15, 15),  # degree
        translation_height_factor=0.2,  # %
        translation_width_factor=0.2,
        zoom_height_factor=0.2,
        zoom_width_factor=0.2,
        shear_height_factor=0.2,
        shear_width_factor=0.2,
        fill_value=114,
        bounding_box_format="xyxy",
    )
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        dataset, bounding_box_format="xyxy", output_path="demo.png"
    )
    print("saved image to ./demo.png")

    # measure performance
    print("measuring augmentation speed...")
    num_iters = 10
    st = time.time()
    dataset_iter = iter(dataset)
    for _ in range(num_iters):
        _ = next(dataset_iter)
    ed = time.time()
    print(f"{1 / ((ed - st) / num_iters) * 9:.3f} FPS")


if __name__ == "__main__":
    main()
