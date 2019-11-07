##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf

# mnist 784
# cifa 3072
class GAN:
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, batch_size, dropout = False, BN = False):

        self.w, self.h, self.ch = input_size
        self.batch_size = batch_size
        self.num_hidden_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_layer_size
        self.start_size = 32
        self.resize_shape = self.hidden_size // 4
        self.X = tf.placeholder(tf.float32, shape = [None, self.w, self.h, self.ch])
        self.Z = tf.placeholder(tf.float32, shape = [None, self.latent_size])

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 0.001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        500, 0.96, staircase=True)

        self.dropout = dropout
        self.BN = BN
        self.dropout_rate = 0.2

    def OutputLayer(self, x, count):
        name = 'l' + str(count)
        with tf.variable_scope(name):
            w = tf.get_variable('w', [self.hidden_size * self.w//4 * self.h//4, 1],
                            dtype=tf.float32 ,initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable('b', [1,],
                            dtype=tf.float32 ,initializer=tf.glorot_uniform_initializer())
            layer = tf.add(tf.matmul(x, w), b)
            prob = tf.nn.sigmoid(layer)
            return layer, prob

    def Flatten(self, x):
        layer = tf.reshape(x, [self.batch_size, -1])
        return layer

    def Conv(self, x, output_dim, stride=1, name="conv2d"):
        with tf.variable_scope(name):
            k = tf.get_variable('kernel', [3, 3, x.get_shape()[-1], output_dim],
                        dtype=tf.float32 ,initializer=tf.glorot_uniform_initializer())
            layer = tf.nn.conv2d(x, k, strides=[1, stride, stride, 1], padding='SAME')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            layer = tf.reshape(tf.nn.bias_add(layer, biases), [self.batch_size] + layer.get_shape().as_list()[1:])
            return layer

    def Deconv(self, x, output_dim, stride=1, name="deconv2d"):
        with tf.variable_scope(name):
            k = tf.get_variable('kernel', [3, 3, output_dim, x.get_shape()[-1]],
                        dtype=tf.float32 ,initializer=tf.glorot_uniform_initializer())
            input_shape = x.get_shape().as_list()
            output_shape = tf.constant([self.batch_size, input_shape[1]*stride, \
                                        input_shape[2]*stride, output_dim])
            layer = tf.nn.conv2d_transpose(x, k, output_shape=output_shape, 
                        strides=[1, stride, stride, 1], padding='SAME')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            layer = tf.reshape(tf.nn.bias_add(layer, biases), [self.batch_size] + layer.get_shape().as_list()[1:])
            return layer

    def ConvBlock(self, x, output_dim, count, stride=1):
        name = 'l' + str(count)
        layer = self.Conv(x, output_dim, stride, name)
        # layer = tf.nn.relu(layer)
        layer = tf.nn.leaky_relu(layer, alpha=0.2)
        return layer

    def DeconvBlock(self, x, output_dim, count, stride=1):
        name = 'l' + str(count)
        layer = self.Deconv(x, output_dim, stride, name)
        # layer = tf.nn.relu(layer)
        layer = tf.nn.leaky_relu(layer, alpha=0.2)
        return layer

    def GatingConv(self, x, output_dim, count, stride=1):
        name = 'l' + str(count)
        layer = self.Deconv(x, output_dim, stride, name)
        prob = tf.nn.sigmoid(layer)
        return layer, prob

    def ResizeLayer(self, x, count):
        name = 'l' + str(count)
        with tf.variable_scope(name):
            w = tf.get_variable('w', [self.latent_size, self.start_size * self.w//4 * self.h//4],
                            dtype=tf.float32 ,initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable('b', [self.start_size  * self.w//4 * self.h//4, ],
                            dtype=tf.float32 ,initializer=tf.glorot_uniform_initializer())
            layer = tf.add(tf.matmul(x, w), b)
            layer = tf.reshape(layer, [-1, self.w//4, self.h//4, self.start_size ])
            # layer = tf.nn.relu(layer)
            layer = tf.nn.leaky_relu(layer, alpha=0.2)        
            return layer

    def Discriminator(self, input_x, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            count = 0
            layer = self.ConvBlock(input_x, 32, count=count)
            count += 1
            layer = self.ConvBlock(layer, self.hidden_size // 2, count=count)
            count += 1
            if self.num_hidden_layers >= 1:
                layer = self.ConvBlock(layer, self.hidden_size // 2, count=count)
                count += 1
            layer = self.ConvBlock(layer, self.hidden_size, count=count, stride=2)
            count += 1
            if self.num_hidden_layers >= 2:
                layer = self.ConvBlock(layer, self.hidden_size, count=count)
                count += 1
            layer = self.ConvBlock(layer, self.hidden_size, count=count, stride=2)
            count += 1
            layer = self.Flatten(layer)
            logits, prob = self.OutputLayer(layer, count=count)
            count += 1
            return tf.squeeze(logits), tf.squeeze(prob)

    def Generator(self, input_z, reuse=False):
        with tf.variable_scope("generator") as scope:
            count = 0
            if reuse:
                scope.reuse_variables()

            layer = self.ResizeLayer(input_z, count=count)
            count += 1
            layer = self.DeconvBlock(layer, self.hidden_size, count=count)
            count += 1
            # if self.num_hidden_layers >= 1:
            #     layer = self.ConvBlock(layer, self.hidden_size, count=count)
            #     count += 1
            layer = self.DeconvBlock(layer, self.hidden_size // 2, count=count, stride=2)
            count += 1
            # if self.num_hidden_layers >= 2:
            #     layer = self.ConvBlock(layer, self.hidden_size // 2, count=count)
            #     count += 1
            layer = self.DeconvBlock(layer, self.hidden_size // 4, count=count, stride=2)
            count += 1    
            logits, output = self.GatingConv(layer, self.ch, count=count)
            count += 1
            return output

    def Generating(self,):
        sample = self.Generator(self.Z, reuse=True)
        return sample

    def Loss(self,):
        G_sample = self.Generator(self.Z)
        D_logit_real, D_real_prob= self.Discriminator(self.X, reuse=False)
        D_logit_fake, D_fake_prob = self.Discriminator(G_sample, reuse=True)
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        D_loss = D_loss_real + D_loss_fake
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

        all_variables = tf.trainable_variables()
        D_variables = [var for var in all_variables if 'discriminator' in var.name]
        G_variables = [var for var in all_variables if 'generator' in var.name]
        return D_loss, G_loss, D_variables, G_variables

    def TrainModel(self, D_loss, G_loss, D_variables, G_variables):
        learning_operation_D = tf.train.AdamOptimizer(learning_rate = self.learning_rate).\
                                minimize(D_loss, global_step = self.global_step, var_list=D_variables)
        learning_operation_G = tf.train.AdamOptimizer(learning_rate = self.learning_rate).\
                                minimize(G_loss, global_step = self.global_step, var_list=G_variables)
        # learning_operation_D = tf.train.AdamOptimizer().minimize(D_loss, var_list=D_variables)
        # learning_operation_G = tf.train.AdamOptimizer().minimize(G_loss, var_list=G_variables)

        return learning_operation_D, learning_operation_G

