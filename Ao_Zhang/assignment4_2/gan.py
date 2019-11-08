import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
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
        self.crossEntropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.gen_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.disc_optimizer = tf.keras.optimizers.Adam(1e-4)

    def Noise(self,):
        return tf.random.normal([self.sample_size, self.latent_size])

    def Generator(self,):
        model = tf.keras.Sequential()
        model.add(layers.Dense(self.w//4*self.h//4*self.hidden_size, use_bias=False, input_shape=(self.latent_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Reshape((self.w//4, self.h//4, self.hidden_size)))
        assert model.output_shape == (None, self.w//4, self.h//4, self.hidden_size)

        model.add(layers.Conv2DTranspose(self.hidden_size//2, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w//4, self.h//4, self.hidden_size//2)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.hidden_size//4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w//2, self.h//2, self.hidden_size//4)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.w, self.h, 1)

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

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.crossEntropy(tf.ones_like(real_output), real_output)
        fake_loss = self.crossEntropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.crossEntropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([BATCH_SIZE, noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))




generated_image = self.generator(noise, training=False)
