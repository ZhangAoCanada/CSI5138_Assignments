import numpy as np
import tensorflow as tf

# mnist 784
# cifa 3072
class GAN:
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
        self.learning_rate_start = 0.0001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        100, 0.96, staircase=True)
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
                        'enc_w_output' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size, self.latent_size))),

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
                        'enc_b_output' : tf.Variable(tf.glorot_uniform_initializer()((self.latent_size,))),

                        'dec_b1' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b2' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b3' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b4' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b5' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b6' : tf.Variable(tf.glorot_uniform_initializer()((self.hidden_layer_size,))),
                        'dec_b_output' : tf.Variable(tf.glorot_uniform_initializer()((self.input_size,))),
                        }

        self.D_variables = [self.weights['enc_w1'], self.weights['enc_w2'], self.weights['enc_w3'],\
                            self.weights['enc_w4'], self.weights['enc_w5'], self.weights['enc_w6'],\
                            self.weights['enc_w_output'], self.biases['enc_b1'], self.biases['enc_b2'],\
                            self.biases['enc_b3'], self.biases['enc_b4'], self.biases['enc_b5'], \
                            self.biases['enc_b6'], self.biases['enc_b_output']]
        self.G_variables = [self.weights['dec_w1'], self.weights['dec_w2'], self.weights['dec_w3'],\
                            self.weights['dec_w4'], self.weights['dec_w5'], self.weights['dec_w6'],\
                            self.weights['dec_w_output'], self.biases['dec_b1'], self.biases['dec_b2'],\
                            self.biases['dec_b3'], self.biases['dec_b4'], self.biases['dec_b5'], \
                            self.biases['dec_b6'], self.biases['dec_b_output']]

    def Discriminator(self, input_x):
        D_weights = []
        D_bias = []
        hidden = tf.nn.relu(tf.add(tf.matmul(input_x, self.weights['enc_w1']), self.biases['enc_b1']))
        for i in range(self.num_hidden_layers):
            w_name = 'enc_w' + str(i + 2)
            b_name = 'enc_b' + str(i + 2)
            hidden = tf.nn.relu(tf.add(tf.matmul(hidden, self.weights[w_name]), self.biases[b_name]))
        logits = tf.add(tf.matmul(hidden, self.weights['enc_w_output']), self.biases['enc_b_output'])   
        prob = tf.nn.sigmoid(logits)
        return logits, prob

    def Generator(self, input_z):
        G_weights = []
        G_bias = []
        hidden = tf.nn.relu(tf.add(tf.matmul(input_z, self.weights['dec_w1']), self.biases['dec_b1']))
        for i in range(self.num_hidden_layers):
            w_name = 'dec_w' + str(i + 2)
            b_name = 'dec_b' + str(i + 2)
            hidden = tf.nn.relu(tf.add(tf.matmul(hidden, self.weights[w_name]), self.biases[b_name]))
        logits = tf.add(tf.matmul(hidden, self.weights['dec_w_output']), self.biases['dec_b_output'])    
        prob = tf.nn.sigmoid(logits)
        return prob

    def Generating(self,):
        sample = self.Generator(self.Z)
        return sample

    def Loss(self, return_loss_only = True):
        G_sample = self.Generator(self.Z)
        D_logit_real, D_real_prob= self.Discriminator(self.X)
        D_logit_fake, D_fake_prob = self.Discriminator(G_sample)
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
        return D_loss, G_loss

    def TrainModel(self):
        D_loss, G_loss = self.Loss()
        learning_operation_D = tf.train.AdamOptimizer(learning_rate = self.learning_rate).\
                                minimize(D_loss, global_step = self.global_step, var_list=self.D_variables)
        learning_operation_G = tf.train.AdamOptimizer(learning_rate = self.learning_rate).\
                                minimize(G_loss, global_step = self.global_step, var_list=self.G_variables)
        return learning_operation_D, learning_operation_G