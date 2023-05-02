import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation._2d.regularization.random_grid_mask import (
    RandomGridMask,
)


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = RandomGridMask(
        size_factor=(0.1, 1.0),
        ratio_factor=(0.5, 0.5),
        rotation_factor=(-5, 5),
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )


if __name__ == "__main__":
    main()
