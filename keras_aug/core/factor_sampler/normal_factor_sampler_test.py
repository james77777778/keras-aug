import tensorflow as tf

import keras_aug


class NormalFactorSamplerTest(tf.test.TestCase):
    def test_sample(self):
        factor = keras_aug.NormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )

        self.assertTrue(0 <= factor() <= 1)

    def test_config(self):
        factor = keras_aug.NormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )

        config = factor.get_config()

        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)

    def test_from_config(self):
        factor = keras_aug.NormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )
        config = factor.get_config()

        factor2 = keras_aug.NormalFactorSampler.from_config(config)
        config = factor2.get_config()

        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)
