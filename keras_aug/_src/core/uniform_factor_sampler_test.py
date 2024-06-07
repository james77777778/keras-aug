from keras.src import testing

from keras_aug._src.core import UniformFactorSampler


class UniformFactorSamplerTest(testing.TestCase):
    def test_sample(self):
        factor = UniformFactorSampler(0.3, 0.6)
        self.assertTrue(0.3 <= factor() <= 0.6)

    def test_config(self):
        factor = UniformFactorSampler(0.3, 0.6)
        config = factor.get_config()
        self.assertEqual(config["lower"], 0.3)
        self.assertEqual(config["upper"], 0.6)

    def test_from_config(self):
        factor = UniformFactorSampler(0.3, 0.6)
        config = factor.get_config()
        factor2 = UniformFactorSampler.from_config(config)
        config = factor2.get_config()
        self.assertEqual(config["lower"], 0.3)
        self.assertEqual(config["upper"], 0.6)
