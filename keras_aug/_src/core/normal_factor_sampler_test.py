from keras.src import testing

from keras_aug._src.core import NormalFactorSampler


class NormalFactorSamplerTest(testing.TestCase):
    def test_sample(self):
        factor = NormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )
        self.assertTrue(0 <= factor() <= 1)

    def test_config(self):
        factor = NormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )
        config = factor.get_config()
        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)

    def test_from_config(self):
        factor = NormalFactorSampler(
            mean=0.5, stddev=0.2, min_value=0, max_value=1
        )
        config = factor.get_config()
        factor2 = NormalFactorSampler.from_config(config)
        config = factor2.get_config()
        self.assertEqual(config["mean"], 0.5)
        self.assertEqual(config["stddev"], 0.2)
