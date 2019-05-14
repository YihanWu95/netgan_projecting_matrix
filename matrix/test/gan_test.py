import logging
import numpy as np
import random
import tempfile
import tensorflow as tf
import unittest

from models import gans
class TestGAN(unittest.TestCase):

    def test_gan(self):
        # This does not belong into a unit test but we keep it here for now to make sure
        # that tf versions are not an issue:
        print(tf.__version__)

        gantest = gans.gan_model()
        gantest.set_up(
            params={
                "latent_shape": 5,
                "original_shape": 10,
                "dis_layer_width": [20,20],
                "gen_layer_width": [20,20],
                "loss_function": "rasgan",
                "gradient_penalty": False,
                "learning_rate" :  1e-4,
                "max_steps": 100,
                "train_ratio": 10
            }
        )
        gantest.train(batch_size = 100, n_step_report = 10)

        return True


if __name__ == '__main__':
    unittest.main()
