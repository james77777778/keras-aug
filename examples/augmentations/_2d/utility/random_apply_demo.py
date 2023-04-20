import time

import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import RandomApply
from keras_aug.augmentations import RandomColorJitter
from keras_aug.augmentations import ResizeAndPad


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=16
    )
    resize = ResizeAndPad(
        height=448, width=448, padding_value=114, bounding_box_format="xyxy"
    )
    augment = RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=(2.0, 2.0),
        contrast_factor=(1.0, 1.0),
        hue_factor=(0.0, 0.0),
        saturation_factor=(1.0, 1.0),
    )
    random_apply = RandomApply(layer=augment, rate=0.75)
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    result = dataset.map(random_apply, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data_across_batch(
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
