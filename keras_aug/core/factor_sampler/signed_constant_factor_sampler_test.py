import tensorflow as tf

import keras_aug


class SignedConstantFactorSamplerTest(tf.test.TestCase):
    def test_sample(self):
        factor = keras_aug.SignedConstantFactorSampler(0.3)
        sample = factor()

        self.assertTrue(sample == 0.3 or sample == -0.3)

    def test_config(self):
        factor = keras_aug.SignedConstantFactorSampler(0.3)

        config = factor.get_config()

        self.assertEqual(config["value"], 0.3)

    def test_from_config(self):
        factor = keras_aug.SignedConstantFactorSampler(0.3)
        config = factor.get_config()

        factor2 = keras_aug.SignedConstantFactorSampler.from_config(config)
        config = factor2.get_config()

        self.assertEqual(config["value"], 0.3)
