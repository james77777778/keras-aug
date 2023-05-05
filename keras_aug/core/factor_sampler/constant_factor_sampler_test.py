import tensorflow as tf

import keras_aug


class ConstantFactorSamplerTest(tf.test.TestCase):
    def test_sample(self):
        factor = keras_aug.ConstantFactorSampler(0.3)

        self.assertEqual(factor(), 0.3)

    def test_config(self):
        factor = keras_aug.ConstantFactorSampler(0.3)

        config = factor.get_config()

        self.assertEqual(config["value"], 0.3)

    def test_from_config(self):
        factor = keras_aug.ConstantFactorSampler(0.3)
        config = factor.get_config()

        factor2 = keras_aug.ConstantFactorSampler.from_config(config)
        config = factor2.get_config()

        self.assertEqual(config["value"], 0.3)
