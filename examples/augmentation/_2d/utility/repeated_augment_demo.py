import tensorflow as tf

from examples import demo_utils
from keras_aug import layers


def main():
    dataset = demo_utils.load_voc_dataset(
        bounding_box_format="xyxy", batch_size=9
    )
    resize = layers.Resize(
        height=224,
        width=224,
        pad_to_aspect_ratio=True,
        padding_value=114,
        bounding_box_format="xyxy",
    )
    layer = layers.RepeatedAugment(
        layers=[
            layers.RandomColorJitter(
                value_range=(0, 255), brightness_factor=(2.0, 2.0)
            ),
            layers.RandomColorJitter(
                value_range=(0, 255), saturation_factor=(0.1, 0.1)
            ),
        ]
    )
    dataset = dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(layer, num_parallel_calls=tf.data.AUTOTUNE)
    demo_utils.visualize_data(
        dataset, bounding_box_format="xyxy", output_path="demo.png"
    )

    # print batch size (x2 after RepeatedAugment)
    dataset_iter = iter(dataset)
    for _ in range(5):
        data = next(dataset_iter)
        print(data["images"].shape[0])


if __name__ == "__main__":
    main()
