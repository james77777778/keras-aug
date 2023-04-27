import time

import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation import RandAugment
from keras_aug.augmentation import Resize


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    resize = Resize(331, 331, bounding_box_format="xyxy")
    layer = RandAugment(
        value_range=(0, 255),
        fill_mode="constant",
        fill_value=128,
        bounding_box_format="xyxy",
    )
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
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
    print(f"cost: {(ed - st) / num_iters:.3f} seconds/batch")


if __name__ == "__main__":
    main()
