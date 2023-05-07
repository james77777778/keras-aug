import time

import tensorflow as tf

from examples import demo_utils
from keras_aug.layers import RandomChoice
from keras_aug.layers import RandomColorJitter
from keras_aug.layers import RandomRotate
from keras_aug.layers import Resize


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=16
    )
    resize = Resize(
        height=224,
        width=224,
        pad_to_aspect_ratio=True,
        padding_value=114,
        bounding_box_format="xyxy",
    )
    augment1 = RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=(2.0, 2.0),
        contrast_factor=(1.0, 1.0),
        hue_factor=(0.0, 0.0),
        saturation_factor=(1.0, 1.0),
    )
    augment2 = RandomRotate(factor=(-10, -10), bounding_box_format="xyxy")
    random_choice = RandomChoice(layers=[augment1, augment2], batchwise=False)
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    result = dataset.map(random_choice, num_parallel_calls=tf.data.AUTOTUNE)
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
