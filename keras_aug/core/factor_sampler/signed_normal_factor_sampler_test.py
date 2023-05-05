import tensorflow as tf

import keras_aug


class SignedNormalFactorSamplerTest(tf.test.TestCase):
    def test_sample(self):
        factor = keras_aug.SignedNormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )

        sample = factor()

        self.assertTrue((0 <= sample <= 1) or (-1 <= sample <= 0))

    def test_sample_nostddev(self):
        factor = keras_aug.SignedNormalFactorSampler(
            mean=0.5, stddev=0.0, min_value=0, max_value=1
        )

        sample = factor()

        self.assertTrue((0 <= sample <= 1) or (-1 <= sample <= 0))

    def test_config(self):
        factor = keras_aug.SignedNormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )

        config = factor.get_config()

        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)

    def test_from_config(self):
        factor = keras_aug.SignedNormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )
        config = factor.get_config()

        factor2 = keras_aug.SignedNormalFactorSampler.from_config(config)
        config = factor2.get_config()

        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)
