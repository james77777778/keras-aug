import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import RandomHSV


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = RandomHSV(
        value_range=(0, 255),
        hue_factor=(-0.015, 0.015),  # 1.5%
        saturation_factor=(0.3, 1.7),  # 70%
        value_factor=(0.6, 1.4),  # 40%
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )

    layer = RandomHSV(
        value_range=(0, 255),
        hue_factor=(0.0, 0.0),
        saturation_factor=(1.0, 1.0),
        value_factor=(1.0, 1.0),
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="no_aug_demo.png"
    )


if __name__ == "__main__":
    main()
