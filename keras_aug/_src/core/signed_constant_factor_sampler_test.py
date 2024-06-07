from keras.src import testing

from keras_aug._src.core import SignedConstantFactorSampler


class SignedConstantFactorSamplerTest(testing.TestCase):
    def test_sample(self):
        factor = SignedConstantFactorSampler(0.3)
        sample = factor()
        self.assertTrue(sample == 0.3 or sample == -0.3)

    def test_config(self):
        factor = SignedConstantFactorSampler(0.3)
        config = factor.get_config()
        self.assertEqual(config["value"], 0.3)

    def test_from_config(self):
        factor = SignedConstantFactorSampler(0.3)
        config = factor.get_config()
        factor2 = SignedConstantFactorSampler.from_config(config)
        config = factor2.get_config()
        self.assertEqual(config["value"], 0.3)
