import time

import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation._2d.auto.rand_augment import RandAugment


def main():
    # slow if bounding boxes are needed
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = RandAugment(
        value_range=(0, 255),
        fill_value=128,
        bounding_box_format="xyxy",
        seed=2023,
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )
    # check performance
    num_iters = 10
    st = time.time()
    for _ in range(num_iters):
        _ = next(iter(result))
    ed = time.time()
    print(f"[NON BATCHWISE] cost: {(ed - st) / num_iters:.3f} seconds/batch")

    # batchwise version, might be faster but the augmentations are all the same
    # within the batch
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = RandAugment(
        value_range=(0, 255),
        fill_value=128,
        batchwise=True,
        bounding_box_format="xyxy",
        seed=2023,
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo_batchwise.png"
    )
    # check performance
    num_iters = 10
    st = time.time()
    for _ in range(num_iters):
        _ = next(iter(result))
    ed = time.time()
    print(f"[BATCHWISE] cost: {(ed - st) / num_iters:.3f} " "seconds/batch")


if __name__ == "__main__":
    main()
