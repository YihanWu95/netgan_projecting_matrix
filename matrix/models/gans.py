import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from typing import Union, Tuple, List

def build_dense_discriminator(
        original_shape: int,
        layer_width: List
):
    '''
    :param original_shape: length of row in up/down projection matrix
    :param layer_width: width of each dense layer
    :return: discriminator network
    '''
    def discriminator(x, reuse=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            discriminator_net = tf.keras.Sequential()
            for i, w in enumerate(layer_width):
                if i==1:
                    discriminator_net.add(tf.keras.layers.Dense(units=w, input_shape=(original_shape,)))
                else:
                    discriminator_net.add(tf.keras.layers.Dense(units=w))

            discriminator_net.add(tf.keras.layers.Dense(units=1))

            logits = discriminator_net(x)
        return logits
    return discriminator


def build_dense_generator(
        latent_shape: int,
        original_shape: int,
        layer_width: List
):
    def generator(x, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            generator_net = tf.keras.Sequential()
            for i, w in enumerate(layer_width):
                if i == 1:
                    generator_net.add(tf.keras.layers.Dense(units=w, input_shape=(latent_shape,)))
                else:
                    generator_net.add(tf.keras.layers.Dense(units=w))

            generator_net.add(tf.keras.layers.Dense(units=original_shape))

            logits = generator_net(x)
        return logits
    return generator


class gan_model():


    def set_up(
            self,
            params: dict,
    ):
        """
        Build WGAN-GP with dense generator and discriminator model.

        :param params: Global model parameters. Must contain:

            - "analytic_kl": bool
            - "original_shape": length of row in up/down projection matrix
            - "dis_layer_width": widths of each layer in discriminator
            - "gen_layer_width": widths of each layer in generator
            - "latent_shape": shape of random vector for generating fake samples
            - "loss_function": wgan-gp, rsgan, rasgan
            - "gradient_penalty" true or false
            - "learning_rate"
            - "max_steps"
            - "train_ratio" ratio of updates between discriminator and generator
        """
        self.train_data = tf.cast(np.load("../train_data.npy"), dtype="float32")
        print(self.train_data.shape)
        self.batch_size = 100
        self.params = params

        # self.labels = np.empty(self.train_data.shape[0])
        # self.batch_size = 100
        # Compute the model loss for discrimator and generator, averaged over
        # the batch size.
        self.discriminator = build_dense_discriminator(
            original_shape = self.params["original_shape"],
            layer_width = self.params["dis_layer_width"]
            # encoder_depth = self.params_enc["encoder_depth"]
        )

        self.generator = build_dense_generator(
            latent_shape = self.params["latent_shape"],
            original_shape=self.params["original_shape"],
            layer_width=self.params["gen_layer_width"]
        )


    def build_input_pipeline(self, batch_size):
        """Build an iterator over training batches."""
        training_dataset = tf.data.Dataset.from_tensor_slices(self.train_data)
        training_batches = training_dataset.shuffle(
            self.train_data.shape[0], reshuffle_each_iteration=True).repeat().batch(batch_size)
        training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)
        true_data = training_iterator.get_next()
        return true_data

    def build_fake_data(self,size):
        """Generate fake samples of proteins."""

        # Generate random noise from a Gaussian distribution.
        fake_images = np.random.normal(size=size)
        return fake_images

    def train(self, batch_size, n_step_report: int = 10):
        self.random_noise = tf.placeholder(tf.float32, shape=[None, self.params["latent_shape"]])
        self.true_data = self.build_input_pipeline(batch_size)
        # print("**************",self.true_proteins.shape, self.labels.shape)
        # self.labels = tf.placeholder(tf.float32, shape=[None, self.func.shape[1]])
        fake_data = self.generator(self.random_noise)
        # fake_protein = tf.squeeze(fake_protein_distribution.sample(1),0)
        print("fake_data", fake_data.shape)

        D_real_logits = self.discriminator(self.true_data)
        D_fake_logits = self.discriminator(fake_data)


        loss_real = tf.reduce_mean(D_real_logits)
        loss_fake = tf.reduce_mean(D_fake_logits)

        #wgan loss function
        if self.params["loss_function"].lower() == "wgan":
            print("Using wgan loss function")
            self.d_loss = loss_fake - loss_real
            self.g_loss =  -loss_fake
        #rsgan loss function
        elif self.params["loss_function"].lower() == "rsgan":
            print("Using rsgan loss function")
            D_real_logits = tf.tile(D_real_logits,[1,tf.shape(D_real_logits)[0]])
            print("shape D_real_logits", D_real_logits.shape)
            diff = D_real_logits - tf.reshape(D_fake_logits,(1,tf.shape(D_real_logits)[0]))
            print("shape diff", diff.shape)
            self.d_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=diff, labels=tf.ones_like(diff)),(0,1))
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=-diff, labels=tf.ones_like(diff)),(0,1))


        elif self.params["loss_function"].lower() == "rasgan":
            print("Using rasgan loss function")
            sig_real = D_real_logits - loss_fake
            sig_fake = D_fake_logits - loss_real
            self.d_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=sig_real, labels=tf.ones_like(sig_real))) + \
                          tf.reduce_mean(
                              tf.nn.sigmoid_cross_entropy_with_logits(logits=sig_fake, labels=tf.zeros_like(sig_fake)))
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=sig_fake, labels=tf.ones_like(sig_fake))) + \
                          tf.reduce_mean(
                              tf.nn.sigmoid_cross_entropy_with_logits(logits=sig_real, labels=tf.zeros_like(sig_real)))
        else:
            raise ValueError("loss_function %s not recognized" % self.params_global["loss_function"])
        """ Gradient Penalty """
        # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
        if self.params["gradient_penalty"]:
            print("Gradient Penalty activiting")
            alpha = tf.random_uniform(shape=tf.shape(self.true_data), minval=0., maxval=1.)
            differences = fake_data - self.true_data  # This is different from MAGAN
            interpolates = self.true_data + (alpha * differences)
            D_inter = self.discriminator(interpolates)
            gradients = tf.gradients(D_inter, [interpolates])[0]
            print("gradients",gradients.shape)
            gradients_sqr = tf.square(gradients)
            slopes = tf.sqrt(tf.reduce_sum(gradients_sqr, reduction_indices=np.arange(1,len(gradients_sqr.shape))))
            print("slopes", slopes.shape)
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            self.lambd = 10  # The higher value, the more stable, but the slower convergence
            self.d_loss += self.lambd * gradient_penalty
        print("XDDDDDDDDDD")
        """ Training """
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.params["learning_rate"]) \
                .minimize(self.d_loss, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'))
            self.g_optim = tf.train.AdamOptimizer(self.params["learning_rate"] * 5) \
                .minimize(self.g_loss, var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'))

        with tf.compat.v1.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            # discriminator pre-train

            for step in range(self.params["max_steps"] + 1):
                # Iterate gradient updates on each network.
                # updata d 10 times before update g
                for _ in range(self.params["train_ratio"]):
                    _, loss_value_d = self.sess.run([self.d_optim, self.d_loss],
                                               feed_dict={self.random_noise: self.build_fake_data(
                                                   [batch_size, self.params["latent_shape"]])})

                _, loss_value_g = self.sess.run([self.g_optim, self.g_loss],
                                           feed_dict={self.random_noise: self.build_fake_data(
                                               [batch_size, self.params["latent_shape"]])})
                # loss_value_d = 0
                if step % n_step_report == 0:
                    # images = self.sess.run(sythetic_image,
                    #                   feed_dict={self.random_noise: self.build_fake_data(
                    #                       [16, self.params_gen["latent_shape"]])})

                    print('Step: {:>3d} Loss_discriminator: {:.3f} '
                          'Loss_generator: {:.3f}'.format(step, loss_value_d, loss_value_g))






