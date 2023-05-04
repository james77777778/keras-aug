import tensorflow as tf

import keras_aug


class UniformFactorSamplerTest(tf.test.TestCase):
    def test_sample(self):
        factor = keras_aug.UniformFactorSampler(0.3, 0.6)

        self.assertTrue(0.3 <= factor() <= 0.6)

    def test_config(self):
        factor = keras_aug.UniformFactorSampler(0.3, 0.6)

        config = factor.get_config()

        self.assertEqual(config["lower"], 0.3)
        self.assertEqual(config["upper"], 0.6)

    def test_from_config(self):
        factor = keras_aug.UniformFactorSampler(0.3, 0.6)
        config = factor.get_config()

        factor2 = keras_aug.UniformFactorSampler.from_config(config)
        config = factor2.get_config()

        self.assertEqual(config["lower"], 0.3)
        self.assertEqual(config["upper"], 0.6)
