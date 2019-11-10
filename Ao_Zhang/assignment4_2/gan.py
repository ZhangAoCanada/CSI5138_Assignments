##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import glob
import numpy as np
from tensorflow.keras import layers
import time

class gan(object):
    def __init__(self, input_size, num_hidden_layers, hidden_layer_size, 
                latent_size, batch_size):
        self.input_size = input_size
        self.w, self.h, self.ch_in = self.input_size
        self.num_layers = num_hidden_layers
        self.latent_size = latent_size
        self.hidden_size = hidden_layer_size
        self.batch_size = batch_size
        self.sample_size = 1

        self.gen = self.Generator()
        self.disc = self.Discriminator()
        self.initial_learning_rate = 1e-4
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                self.initial_learning_rate,
                                                decay_steps=2000,
                                                decay_rate=0.96,
                                                staircase=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    def Noise(self,):
        return tf.random.normal([self.batch_size, self.latent_size])

    def Generator(self,):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.w//8*self.h//8*self.hidden_size, use_bias=False, input_shape=(self.latent_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((self.w//8, self.h//8, self.hidden_size)))

        model.add(layers.Conv2DTranspose(self.hidden_size//2, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.hidden_size//4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.ch_in, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

        return model

    def Discriminator(self,):
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(self.hidden_size//4, (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[self.w, self.h, self.ch_in]))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(self.hidden_size//2, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv2D(self.hidden_size, (5, 5), strides=(2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def DiscriminatorLoss(self, real_output, fake_output):
        real_loss = self.crossEntropy(tf.ones_like(real_output), real_output)
        fake_loss = self.crossEntropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def GeneratorLoss(self, fake_output):
        return self.crossEntropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def Training(self, images):
        noise = self.Noise()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gen(noise, training=True)

            real_output = self.disc(images, training=True)
            fake_output = self.disc(generated_images, training=True)

            gen_loss = self.GeneratorLoss(fake_output)
            disc_loss = self.DiscriminatorLoss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc.trainable_variables))

        return gen_loss, disc_loss


