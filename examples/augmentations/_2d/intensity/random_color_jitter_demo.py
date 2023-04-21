import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation import RandomColorJitter


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = RandomColorJitter(
        value_range=(0, 255),
        brightness_factor=(0.6, 1.4),  # 40%
        contrast_factor=(0.6, 1.4),  # 40%
        hue_factor=(-0.015, 0.015),  # 1.5%
        saturation_factor=(0.6, 1.4),  # 40%
    )
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )


if __name__ == "__main__":
    main()
