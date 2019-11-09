##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import glob
import numpy as np
from tensorflow.keras import layers
import time


class vae(tf.keras.Model):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, batch_size):
        super(vae, self).__init__()
        self.input_size = input_size
        self.w, self.h, self.ch_in = self.input_size
        self.num_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_layer_size
        self.batch_size = batch_size
        self.sample_size = 1

        self.dec = self.Decoder()
        self.enc = self.Encoder()
        self.initial_learning_rate = 1e-3
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='sum')
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                self.initial_learning_rate,
                                                decay_steps=100000,
                                                decay_rate=0.96,
                                                staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.trainable_var = self.dec.trainable_variables + self.enc.trainable_variables

    def SampleLatent(self, mean, variance):
        eps = tf.random.normal(shape = tf.shape(mean))
        return mean + tf.exp(variance / 2) * eps

    def Decoder(self,):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.w//4*self.h//4*self.hidden_size, use_bias=False, input_shape=(self.latent_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((self.w//4, self.h//4, self.hidden_size)))
        assert model.output_shape == (None, self.w//4, self.h//4, self.hidden_size)

        model.add(layers.Conv2DTranspose(self.hidden_size//2, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w//4, self.h//4, self.hidden_size//2)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if self.num_layers >= 1:
            model.add(layers.Conv2DTranspose(self.hidden_size//2, (3, 3), strides=(1, 1), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.hidden_size//4, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w//2, self.h//2, self.hidden_size//4)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if self.num_layers >= 2:
            model.add(layers.Conv2DTranspose(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.ch_in, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w, self.h, self.ch_in)

        return model

    def Encoder(self,):
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same',
                                        input_shape=[self.w, self.h, self.ch_in]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        if self.num_layers >= 1:
            model.add(layers.Conv2D(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(self.hidden_size//2, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        if self.num_layers >= 2:
            model.add(layers.Conv2D(self.hidden_size//2, (3, 3), strides=(1, 1), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(self.hidden_size, (3, 3), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        if self.num_layers >= 3:
            model.add(layers.Conv2D(self.hidden_size, (3, 3), strides=(1, 1), padding='same'))
            model.add(layers.LeakyReLU())
            model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(2 * self.latent_size))

        return model
    
    def Encoding(self, x):
        mean, logvar = tf.split(self.enc(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def Reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def Decoding(self, z, apply_sigmoid=False):
        logits = self.dec(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def LogNormalPdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum( -.5 * ((sample - mean) ** 2. \
                            * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    # @tf.function
    def Loss(self, x):
        mean, logvar = self.Encoding(x)
        z = self.Reparameterize(mean, logvar)
        x_logit = self.Decoding(z)

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
        logpz = self.LogNormalPdf(z, 0., 0.)
        logqz_x = self.LogNormalPdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def Training(self, x):
        with tf.GradientTape() as tape:
            loss = self.Loss(x)
        gradients = tape.gradient(loss, self.trainable_var)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_var))
        return loss

