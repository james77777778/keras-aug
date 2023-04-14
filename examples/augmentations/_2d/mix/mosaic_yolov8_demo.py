import time

import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import MosaicYOLOV8
from keras_aug.augmentations import ResizeLongest


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    resize = ResizeLongest(max_size=320, bounding_box_format="xyxy")
    mosaic = MosaicYOLOV8(
        height=640,
        width=640,
        padding_value=114,
        bounding_box_format="xyxy",
    )
    result = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    result = result.map(mosaic, num_parallel_calls=tf.data.AUTOTUNE)
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