##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf

# mnist 784
# cifa 3072
class VAE:
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, dropout = False, BN = False):
        assert isinstance(input_size, int)
        assert isinstance(num_hidden_layers, int)
        assert isinstance(latent_size, int)
        if num_hidden_layers > 5:
            raise ValueError("Maximum 5 hidden layers, no more.")
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_layer_size = hidden_layer_size
        self.X = tf.placeholder(tf.float32, shape = [None, self.input_size])
        self.Z = tf.placeholder(tf.float32, shape = [None, self.latent_size])

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 0.001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        500, 0.96, staircase=True)
        self.dropout = dropout
        self.BN = BN
        self.dropout_rate = 0.2

        self.weights = {
                        'enc_w1' : tf.Variable(tf.glorot_uniform_initializer()((self.input_size, self.hidden_layer_size))),
                        'enc_w2' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'enc_w3' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'enc_w4' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'enc_w5' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'enc_w6' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'enc_w_mu' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.latent_size))),
                        'enc_w_var' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.latent_size))),

                        'dec_w1' : tf.Variable(tf.glorot_uniform_initializer()((self.latent_size, self.hidden_layer_size))),
                        'dec_w2' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'dec_w3' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'dec_w4' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'dec_w5' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'dec_w6' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.hidden_layer_size))),
                        'dec_w_output' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.input_size))),
                        }
        self.biases = {
                        'enc_b1' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'enc_b2' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'enc_b3' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'enc_b4' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'enc_b5' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'enc_b6' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'enc_b_mu' : tf.Variable(tf.glorot_uniform_initializer()((self.latent_size,))),
                        'enc_b_var' : tf.Variable(tf.glorot_uniform_initializer()((self.latent_size,))),

                        'dec_b1' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b2' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b3' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b4' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b5' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b6' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b_output' : tf.Variable(tf.glorot_uniform_initializer()((self.input_size,))),
                        }

    def Encoder(self):
        hidden = tf.nn.relu(tf.add(tf.matmul(self.X, self.weights['enc_w1']), self.biases['enc_b1']))
        if self.num_hidden_layers != 0:
            for i in range(self.num_hidden_layers):
                w_name = 'enc_w' + str(i + 2)
                b_name = 'enc_b' + str(i + 2)
                hidden = tf.nn.relu(tf.add(tf.matmul(hidden, self.weights[w_name]), self.biases[b_name]))
        mean = tf.add(tf.matmul(hidden, self.weights['enc_w_mu']), self.biases['enc_b_mu'])   
        variance = tf.add(tf.matmul(hidden, self.weights['enc_w_var']), self.biases['enc_b_var'])        
        return mean, variance

    def Decoder(self, x_input):
        hidden = tf.nn.relu(tf.add(tf.matmul(x_input, self.weights['dec_w1']), self.biases['dec_b1']))
        if self.num_hidden_layers != 0:
            for i in range(self.num_hidden_layers):
                w_name = 'dec_w' + str(i + 2)
                b_name = 'dec_b' + str(i + 2)
                hidden = tf.nn.relu(tf.add(tf.matmul(hidden, self.weights[w_name]), self.biases[b_name]))
        logits = tf.add(tf.matmul(hidden, self.weights['dec_w_output']), self.biases['dec_b_output'])    
        prob = tf.nn.sigmoid(logits)
        return logits, prob

    def Generating(self,):
        random_logits, random_prob = self.Decoder(self.Z)
        return random_prob

    def SampleLatent(self, mean, variance):
        eps = tf.random.normal(shape = tf.shape(mean))
        return mean + tf.exp(variance / 2) * eps

    def Loss(self,):
        mean, variance = self.Encoder()
        sampling = self.SampleLatent(mean, variance)
        sample_logits, sample_prob = self.Decoder(sampling)
        # E(log p|z (x_i | z_i))
        reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=sample_logits, labels=self.X), axis=1)
        # KL(q z|x (.|x_i) || p Z), since the variance we got is actually log(variance)
        # therefore, we move everthing inside log
        KL_loss = 0.5 * tf.reduce_sum(tf.exp(variance) + tf.square(mean) - 1. - variance, axis=1)
        # vae loss function
        vae_loss = tf.reduce_mean(reconstruction_loss + KL_loss)
        return vae_loss

    def TrainModel(self,):
        loss = self.Loss()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        return learning_operation




