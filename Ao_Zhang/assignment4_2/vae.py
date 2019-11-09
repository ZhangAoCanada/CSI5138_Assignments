##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import glob
import numpy as np
from tensorflow.keras import layers
import time


class vae(object):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, batch_size):
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

        model.add(layers.Conv2DTranspose(self.hidden_size//4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w//2, self.h//2, self.hidden_size//4)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        if self.num_layers >= 2:
            model.add(layers.Conv2DTranspose(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same', use_bias=False))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.ch_in, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.w, self.h, self.ch_in)

        return model

    def Encoder(self,):
        X = tf.keras.Input(shape = [self.w, self.h, self.ch_in])

        hidden = layers.Conv2D(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same')(X)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Dropout(0.3)(hidden)

        if self.num_layers >= 1:
            hidden = layers.Conv2D(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same')(hidden)
            hidden = layers.LeakyReLU()(hidden)
            hidden = layers.Dropout(0.3)(hidden)

        hidden = layers.Conv2D(self.hidden_size//2, (5, 5), strides=(2, 2), padding='same')(hidden)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Dropout(0.3)(hidden)

        if self.num_layers >= 2:
            hidden = layers.Conv2D(self.hidden_size//2, (3, 3), strides=(1, 1), padding='same')(hidden)
            hidden = layers.LeakyReLU()(hidden)
            hidden = layers.Dropout(0.3)(hidden)

        hidden = layers.Conv2D(self.hidden_size, (5, 5), strides=(2, 2), padding='same')(hidden)
        hidden = layers.LeakyReLU()(hidden)
        hidden = layers.Dropout(0.3)(hidden)

        if self.num_layers >= 3:
            hidden = layers.Conv2D(self.hidden_size, (3, 3), strides=(1, 1), padding='same')(hidden)
            hidden = layers.LeakyReLU()(hidden)
            hidden = layers.Dropout(0.3)(hidden)

        hidden = layers.Flatten()(hidden)

        mean = layers.Dense(self.latent_size)(hidden)
        variance = layers.Dense(self.latent_size)(hidden)

        return tf.keras.Model(X, [mean, variance])

    def ReconstructionLoss(self, production, real):
        # prod = layers.Flatten()(production)
        # r = layers.Flatten()(real)
        real_loss = self.crossEntropy(real, production)
        return real_loss

    def KLDLoss(self, mean, variance):
        KL_loss = 0.5 * tf.reduce_sum(tf.exp(variance) + tf.square(mean) - 1. - variance, axis=1)
        return tf.reduce_sum(KL_loss)

    @tf.function
    def Training(self, images):
        
        with tf.GradientTape() as tape:
            mean, variance = self.enc(images)

            sample_latent = self.SampleLatent(mean, variance)
            produced_img = self.dec(sample_latent)

            reconstruct_loss = self.ReconstructionLoss(produced_img, images)
            kl_loss = self.KLDLoss(mean, variance)
            
            total_loss = reconstruct_loss + kl_loss

            all_variables = self.dec.trainable_variables + self.enc.trainable_variables

        gradients = tape.gradient(total_loss, all_variables)

        self.optimizer.apply_gradients(zip(gradients, all_variables))

        return total_loss


