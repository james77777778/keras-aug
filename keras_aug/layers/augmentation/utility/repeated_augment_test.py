import tensorflow as tf

from keras_aug import layers


class RepeatedAugmentationTest(tf.test.TestCase):
    def test_output_shapes(self):
        repeated_augment = layers.RepeatedAugment(
            layers=[
                layers.RandAugment(value_range=(0, 255)),
                layers.RandomFlip(),
            ]
        )
        inputs = {
            "images": tf.ones((4, 8, 8, 3)),
            "labels": tf.ones((4,)),
        }
        outputs = repeated_augment(inputs)

        self.assertEqual(outputs["images"].shape, (8, 8, 8, 3))
        self.assertEqual(outputs["labels"].shape, (8,))

    def test_with_mix_up(self):
        repeated_augment = layers.RepeatedAugment(
            layers=[
                layers.RandAugment(value_range=(0, 255)),
                layers.MixUp(),
            ]
        )
        inputs = {
            "images": tf.ones((4, 8, 8, 3)),
            "labels": tf.ones((4, 10)),
        }
        outputs = repeated_augment(inputs)

        self.assertEqual(outputs["images"].shape, (8, 8, 8, 3))
        self.assertEqual(outputs["labels"].shape, (8, 10))
