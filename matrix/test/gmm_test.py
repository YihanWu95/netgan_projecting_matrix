from models import gmm
import unittest

class TestVAE(unittest.TestCase):

    def test_vae(self):
        gmmtest = gmm.gmm_model()
        gmmtest.set_up()
        gmmtest.train()