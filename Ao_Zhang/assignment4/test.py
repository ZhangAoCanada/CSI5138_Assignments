##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
import numpy as np
import tensorflow as tf
from vae import VAE
from GAN import GAN
from WGAN import WGAN
from mlxtend.data import loadlocal_mnist
import pickle
from tqdm import tqdm

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

##################################################################
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

    # find how many classes
    all_classes = np.unique(Y_train_original)
    num_class = len(all_classes) 
    num_input = X_train.shape[1] 

    # transfer label format
    Y_train = TranslateLables(Y_train_original, num_class)
    Y_test = TranslateLables(Y_test_original, num_class)
    return X_train, Y_train, X_test, Y_test, num_input, num_class

##################################################################
def ReadCFAR(file_name, data_file=True):
    with open(file_name, 'rb') as fo:
        cfar_dict = pickle.load(fo, encoding='bytes')

    if data_file:
        data = cfar_dict[b'data']
        filenames = cfar_dict[b'filenames']
        labels = cfar_dict[b'labels']
        batch_label = cfar_dict[b'batch_label']
        return data
    else:
        label_names = cfar_dict[b'label_names']
        num_vis = cfar_dict[b'num_vis']
        num_cases_per_batch = cfar_dict[b'num_cases_per_batch']
        return num_cases_per_batch



# cfar_dir = "cifar-10-batches-py/"
# cfar_filename = cfar_dir + "data_batch_1"
# cfar_filename_catlog = cfar_dir + "batches.meta"

# cfar_data = ReadCFAR(cfar_filename, True)
# # cfar_data = ReadCFAR(cfar_filename_catlog, False)

# print(cfar_data.shape)


Mnist_local_path = "mnist/"
X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path)

batch_size = 500
hm_batches_train = len(X_train) // batch_size


test_data = X_train[:1] / 255.

def SampleZ(size):
    return np.random.uniform(-1., 1., size=size)


# model = VAE(784, 3, 256, 20)
model = GAN(784, 1, 256, 20)
# model = WGAN(784, 1, 256, 20)

example = model.Generating()

loss_D, loss_G = model.Loss()
train_op_D, train_op_G = model.TrainModel()
# clip_D = model.ClipDiscriminatorWeights()

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(allow_growth=True)


fig = plt.figure()
ax = fig.add_subplot(111)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for i in tqdm(range(10000000)):
        for each_batch_train in range(hm_batches_train):
            X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
            X_train_batch /= 255.
            for whatever in range(5):
                # _, Dloss, _ = sess.run([train_op_D, loss_D, clip_D], feed_dict={model.X: X_train_batch, model.Z: SampleZ([10, 20])})
                _, Dloss = sess.run([train_op_D, loss_D], feed_dict={model.X: X_train_batch, model.Z: SampleZ([10, 20])})
            _, Gloss = sess.run([train_op_G, loss_G], feed_dict={model.X: X_train_batch, model.Z: SampleZ([10, 20])})
            if (i*batch_size + each_batch_train) % 100 == 0:
                print([Dloss, Gloss])

                sample = sess.run(example, feed_dict={model.Z: SampleZ([1, 20])})
                sample = sample.reshape((28, 28))
                gt = test_data[0].reshape((28, 28))
                plt.cla()
                ax.clear()
                ax.imshow(np.concatenate([gt, sample], axis = 1))
                fig.canvas.draw()
                plt.pause(0.01)

