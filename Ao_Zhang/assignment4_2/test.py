##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import glob
import imageio
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
from tensorflow.keras import layers
import time
import pickle
from tqdm import tqdm
from mlxtend.data import loadlocal_mnist

from vae import vae
from gan import gan
from wgan import wgan

############################## MNIST LOADING ############################
def TranslateLables(labels, num_class):
    """
    Function:
        Transfer ground truth labels to one hot format, e.g.
        1       ->      [0, 1, 0, 0, 0, ..., 0]
    """
    hm_labels = len(labels)
    label_onehot = np.zeros((hm_labels, num_class), dtype = np.float32)
    for i in range(hm_labels):
        current_class = labels[i]
        label_onehot[i, current_class] = 1.
    return label_onehot

def GetMnistData(data_path):
    """
    Function:
        Read mnist dataset and transfer it into wanted format.
        For input:
            if not CNN: (60000, 784)
            elif CNN:   (60000, 28, 28, 1)
        For output: one hot
            [0, 1, 0, 0, ..., 0]
    """
    # read dataset
    X_train, Y_train_original = loadlocal_mnist(
            images_path=data_path + "train-images-idx3-ubyte", 
            labels_path=data_path + "train-labels-idx1-ubyte")
    X_test, Y_test_original = loadlocal_mnist(
            images_path=data_path + "t10k-images-idx3-ubyte", 
            labels_path=data_path + "t10k-labels-idx1-ubyte")
    # transfer into float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train /= 255.
    X_test /= 255.

    # find how many classes
    all_classes = np.unique(Y_train_original)
    num_class = len(all_classes) 
    num_input = X_train.shape[1] 

    # transfer label format
    Y_train = TranslateLables(Y_train_original, num_class)
    Y_test = TranslateLables(Y_test_original, num_class)
    return X_train, Y_train, X_test, Y_test, num_input, num_class

############################## CIFAR LOADING ############################
def ReadCifarLabels(cfar_dir):
    file_name = cfar_dir + "batches.meta"
    with open(file_name, 'rb') as fo:
        cfar_dict = pickle.load(fo, encoding='bytes')

    label_names = cfar_dict[b'label_names']
    num_vis = cfar_dict[b'num_vis']
    num_cases_per_batch = cfar_dict[b'num_cases_per_batch']
    return label_names

def ReadCifarData(file_name):
    with open(file_name, 'rb') as fo:
        cfar_dict = pickle.load(fo, encoding='bytes')
    data = cfar_dict[b'data']
    filenames = cfar_dict[b'filenames']
    labels = cfar_dict[b'labels']
    batch_label = cfar_dict[b'batch_label']
    return data, labels, data.shape[1]

def LoadAllCifarData(cfar_dir):
    batch_file_1 = cfar_dir + "data_batch_1"
    batch_file_2 = cfar_dir + "data_batch_2"
    batch_file_3 = cfar_dir + "data_batch_3"
    batch_file_4 = cfar_dir + "data_batch_4"
    batch_file_5 = cfar_dir + "data_batch_5"
    batch_file_test = cfar_dir + "test_batch"
    all_batch_train = [batch_file_1, batch_file_2, batch_file_3, batch_file_4, batch_file_5]
    X_train = []
    Y_train = []
    for each_file in all_batch_train:
        data, labels, input_size = ReadCifarData(each_file)
        X_train.append(data)
        Y_train.append(labels)
    X_train = np.concatenate(X_train, axis = 0).astype(np.float32) / 255.
    Y_train = np.concatenate(Y_train, axis = 0).astype(np.float32)
    X_test, Y_test, input_size = ReadCifarData(batch_file_test)
    X_test = X_test.astype(np.float32) / 255.
    Y_test = np.array(Y_test)
    Y_test = Y_test.astype(np.float32)
    return X_train, Y_train, X_test, Y_test, input_size

def TransferToImage(data):
    r_channel = data[:, :1024].reshape((-1, 32, 32, 1))
    g_channel = data[:, 1024:2048].reshape((-1, 32, 32, 1))
    b_channel = data[:, 2048:].reshape((-1, 32, 32, 1))
    imgs = np.concatenate([r_channel, g_channel, b_channel], axis = -1)
    return imgs

def FormSamples(dataset_name, current_samples):
    if dataset_name == "MNIST":
        current_samples = current_samples.reshape((10, 20, 28, 28))
    else:
        current_samples = current_samples.reshape((10, 20, 32, 32, 3))

    all_imgs = []
    for i in range(3):
        row_imgs = []
        for j in range(10):
            row_imgs.append(current_samples[i, j])
        row_imgs = np.concatenate(row_imgs, axis = 1)
        all_imgs.append(row_imgs)
    all_imgs = np.concatenate(all_imgs, axis = 0)
    return all_imgs


cifar_dir = "cifar-10-batches-py/"
X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)
X_train = TransferToImage(X_train)

fig = plt.figure()

ax = fig.add_subplot(111)

for i in range(len(X_train)):
    current_X = X_train[i]

    plt.cla()
    ax.clear()
    ax.imshow(current_X)
    fig.canvas.draw()
    plt.pause(0.01)

