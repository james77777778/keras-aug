import tensorflow as tf

from examples import demo_utils
from keras_aug.augmentation import CutMix
from keras_aug.augmentation import ResizeAndPad


def main():
    dataset = demo_utils.load_oxford_dataset()
    resize = ResizeAndPad(height=224, width=224)
    mix_up = CutMix(alpha=1.0)
    result = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    result = result.map(mix_up, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        result, bounding_box_format=None, output_path="demo.png"
    )


if __name__ == "__main__":
    main()
