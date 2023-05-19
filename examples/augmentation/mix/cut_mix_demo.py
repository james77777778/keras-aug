import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from keras_aug.layers import CutMix
from keras_aug.layers import Resize
from keras_aug.utils import demo as demo_utils


def main():
    dataset = demo_utils.load_oxford_dataset()
    resize = Resize(224, 224, crop_to_aspect_ratio=True)
    layer = CutMix()
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        dataset, bounding_box_format=None, output_path="demo.png"
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

    # show segmentation masks
    dataset = demo_utils.load_oxford_iiit_pet_dataset()
    resize = Resize(224, 224, crop_to_aspect_ratio=True)
    layer = CutMix()
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_segmentation_masks(
        dataset,
        image_value_range=(0, 255),
        mask_value_range=(0, 2),
        output_path="demo_mask.png",
    )
    print("saved image to ./demo_masks.png")


if __name__ == "__main__":
    main()
