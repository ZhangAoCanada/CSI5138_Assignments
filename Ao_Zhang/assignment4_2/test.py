    # def Generator(self,):
    #     model = tf.keras.Sequential()

    #     if self.dataset_name == "CIFAR":
    #         model.add(layers.Dense(self.w//16*self.h//16*self.hidden_size, input_shape=(self.latent_size,), use_bias=False))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))

    #         model.add(layers.Reshape((self.w//16, self.h//16, self.hidden_size)))

    #         if self.num_layers >= 1:
    #             model.add(layers.Conv2DTranspose(self.hidden_size, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
    #             model.add(layers.BatchNormalization())
    #             model.add(layers.LeakyReLU(alpha=0.2))

    #         model.add(layers.Conv2DTranspose(self.hidden_size, (self.k_s, self.k_s), strides=(2, 2), padding='same', use_bias=False))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))

    #         if self.num_layers >= 2:
    #             model.add(layers.Conv2DTranspose(self.hidden_size//2, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
    #             model.add(layers.BatchNormalization())
    #             model.add(layers.LeakyReLU(alpha=0.2))

    #         model.add(layers.Conv2DTranspose(self.hidden_size//2, (self.k_s, self.k_s), strides=(2, 2), padding='same', use_bias=False))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))
    #     else:
    #         model.add(layers.Dense(self.w//4*self.h//4*self.hidden_size, input_shape=(self.latent_size,), use_bias=False))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))

    #         model.add(layers.Reshape((self.w//4, self.h//4, self.hidden_size)))

    #         if self.num_layers >= 1:
    #             model.add(layers.Conv2DTranspose(self.hidden_size//2, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
    #             model.add(layers.BatchNormalization())
    #             model.add(layers.LeakyReLU(alpha=0.2))

    #     model.add(layers.Conv2DTranspose(self.hidden_size//4, (self.k_s, self.k_s), strides=(2, 2), padding='same', use_bias=False))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.LeakyReLU(alpha=0.2))

    #     if self.num_layers >= 3:
    #         model.add(layers.Conv2DTranspose(self.hidden_size//4, (self.k_s, self.k_s), strides=(1, 1), padding='same', use_bias=False))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))

    #     model.add(layers.Conv2DTranspose(self.ch_in, (self.k_s, self.k_s), strides=(2, 2), padding='same', activation='tanh', use_bias=False))

    #     return model

    # def Discriminator(self,):
    #     model = tf.keras.Sequential()

    #     if self.dataset_name == "CIFAR":
    #         model.add(layers.Conv2D(self.hidden_size//4, (self.k_s, self.k_s), strides=(2, 2), padding='same',
    #                                         input_shape=[self.w, self.h, self.ch_in]))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))
    #         # model.add(layers.Dropout(0.2))

    #         if self.num_layers >= 1:
    #             model.add(layers.Conv2D(self.hidden_size//4, (self.k_s, self.k_s), strides=(1, 1), padding='same'))
    #             model.add(layers.BatchNormalization())
    #             model.add(layers.LeakyReLU(alpha=0.2))
    #             # model.add(layers.Dropout(0.2))

    #         model.add(layers.Conv2D(self.hidden_size//2, (self.k_s, self.k_s), strides=(2, 2), padding='same'))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))
    #         # model.add(layers.Dropout(0.2))

    #         if self.num_layers >= 2:
    #             model.add(layers.Conv2D(self.hidden_size//2, (self.k_s, self.k_s), strides=(1, 1), padding='same'))
    #             model.add(layers.BatchNormalization())
    #             model.add(layers.LeakyReLU(alpha=0.2))
    #             # model.add(layers.Dropout(0.2))

    #         model.add(layers.Conv2D(self.hidden_size, (self.k_s, self.k_s), strides=(2, 2), padding='same'))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))
    #         # model.add(layers.Dropout(0.2))
    #     else:
    #         model.add(layers.Conv2D(self.hidden_size//2, (self.k_s, self.k_s), strides=(2, 2), padding='same',
    #                                         input_shape=[self.w, self.h, self.ch_in]))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))
    #         # model.add(layers.Dropout(0.2))

    #         if self.num_layers >= 1:
    #             model.add(layers.Conv2D(self.hidden_size//2, (self.k_s, self.k_s), strides=(1, 1), padding='same'))
    #             model.add(layers.BatchNormalization())
    #             model.add(layers.LeakyReLU(alpha=0.2))
    #             # model.add(layers.Dropout(0.2))

    #     model.add(layers.Conv2D(self.hidden_size, (self.k_s, self.k_s), strides=(2, 2), padding='same'))
    #     model.add(layers.BatchNormalization())
    #     model.add(layers.LeakyReLU(alpha=0.2))
    #     # model.add(layers.Dropout(0.2))

    #     if self.num_layers >= 3:
    #         model.add(layers.Conv2D(self.hidden_size, (self.k_s, self.k_s), strides=(1, 1), padding='same'))
    #         model.add(layers.BatchNormalization())
    #         model.add(layers.LeakyReLU(alpha=0.2))
    #         # model.add(layers.Dropout(0.2))

    #     model.add(layers.Flatten())
    #     model.add(layers.Dense(1))

    #     return model