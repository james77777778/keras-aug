import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentations import ChannelDropout


def main():
    dataset = demo_utils.load_voc_dataset(bounding_box_format="xyxy")
    layer = ChannelDropout()
    result = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format="xyxy", output_path="demo.png"
    )


if __name__ == "__main__":
    main()
