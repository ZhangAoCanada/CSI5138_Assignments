import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np



a = tf.ones([1, 10, 10, 3])

def Encoder():
    X = tf.keras.Input(shape = [10, 10, 3])

    hidden = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(X)
    hidden = layers.LeakyReLU()(hidden)
    hidden = layers.Dropout(0.3)(hidden)

    hidden = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(hidden)
    hidden = layers.LeakyReLU()(hidden)
    hidden = layers.Dropout(0.3)(hidden)

    hidden = layers.Flatten()(hidden)

    mean = layers.Dense(10)(hidden)
    variance = layers.Dense(10)(hidden)

    return tf.keras.Model(X, [mean, variance])

model = Encoder()

print(len(model.trainable_variables))

test1, test2 = model(a)

print(test1.shape)
print(test2.shape)
