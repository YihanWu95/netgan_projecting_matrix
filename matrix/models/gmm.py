import tensorflow as tf
import numpy as np
from tensorflow.python.framework import constant_op

# train_data = np.load("../train_data.npy")
# print(train_data.shape)


class gmm_model():

    def set_up(self):
        self.train_data = tf.cast(np.load("../train_data.npy"),dtype="float32")
        print(self.train_data.shape)
        self.batch_size = 100
        self.num_centers = 2
        self.steps = 10000
        self.max_steps = 100
        


    def input_fn(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        dataset = tf.data.Dataset.from_tensor_slices(self.train_data).batch(batch_size).repeat()
        return tf.compat.v1.data.make_one_shot_iterator(dataset).get_next(),None


    def train(self):
        gmm = tf.contrib.factorization.GMM(num_clusters = self.num_centers,
                                           model_dir=None,
                                           random_seed=0,
                                           params='wmc',
                                           initial_clusters='random',
                                           covariance_type='full',
                                           config=None)
        # for _ in range(10):
        gmm.fit(input_fn = self.input_fn, steps=self.steps)
        clusters = gmm.clusters()
        print("clusters",clusters.shape)
        cov = gmm.covariances()
        print("covariance", cov.shape)
        # print(clusters)