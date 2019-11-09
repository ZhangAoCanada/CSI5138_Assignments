##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import glob
import numpy as np
from tensorflow.keras import layers
import time

class wgan(object):
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
                                                decay_steps=10000,
                                                decay_rate=0.96,
                                                staircase=True)
        self.gen_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        self.disc_optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)

    def Noise(self,):
        return tf.random.normal([self.batch_size, self.latent_size])

    def Generator(self,):
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

        model.add(layers.Conv2DTranspose(self.hidden_size//4, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, self.w//2, self.h//2, self.hidden_size//4)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())

        model.add(layers.Conv2DTranspose(self.ch_in, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, self.w, self.h, self.ch_in)

        return model

    def Discriminator(self,):
        model = tf.keras.Sequential()

        model.add(layers.Conv2D(self.hidden_size//4, (3, 3), strides=(1, 1), padding='same',
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
        loss = tf.reduce_mean(real_output) - tf.reduce_mean(fake_output)
        return loss

    def GeneratorLoss(self, fake_output):
        return - tf.reduce_mean(fake_output)

    @tf.function
    def Training(self, images):
        noise = self.Noise()

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.gen(noise, training=True)

            real_output = self.disc(images, training=True)
            fake_output = self.disc(generated_images, training=True)

            disc_loss = - self.DiscriminatorLoss(real_output, fake_output)
            gen_loss = self.GeneratorLoss(fake_output)

        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.disc.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

        for _ in range(5):
            self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.disc.trainable_variables))
            self.ClipDiscWeights()

        self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.gen.trainable_variables))

        return gen_loss, disc_loss

    def ClipDiscWeights(self):
        for var in self.disc.trainable_variables:
            var.assign(tf.clip_by_value(var, -0.01, 0.01))


