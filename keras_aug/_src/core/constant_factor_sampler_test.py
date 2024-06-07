from keras.src import testing

from keras_aug._src.core import ConstantFactorSampler


class ConstantFactorSamplerTest(testing.TestCase):
    def test_sample(self):
        factor = ConstantFactorSampler(0.3)
        self.assertEqual(factor(), 0.3)

    def test_config(self):
        factor = ConstantFactorSampler(0.3)
        config = factor.get_config()
        self.assertEqual(config["value"], 0.3)

    def test_from_config(self):
        factor = ConstantFactorSampler(0.3)
        config = factor.get_config()
        factor2 = ConstantFactorSampler.from_config(config)
        config = factor2.get_config()
        self.assertEqual(config["value"], 0.3)
