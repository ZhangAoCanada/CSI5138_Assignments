import numpy as np
from VAE import VAE
from GAN import GAN
from WGAN import WGAN
from mlxtend.data import loadlocal_mnist
import pickle
from tqdm import tqdm

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

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

cifar_dir = "cifar-10-batches-py/"
X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)

print(X_train.max())

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(len(X_train)):
    test = X_train[i:i+1]
    imgs = TransferToImage(test)
    plt.cla()
    ax.clear()
    ax.imshow(imgs[0])
    fig.canvas.draw()
    plt.pause(0.01)
