##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
import numpy as np
import tensorflow as tf
from vae import VAE
from mlxtend.data import loadlocal_mnist
import pickle

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

test_data = X_train[:10] / 255.


model = VAE(784, 3, 100)
loss = model.Loss()
train_op = model.TrainModel()

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(allow_growth=True)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for i in range(10000):
        _, t1 = sess.run([train_op, loss], feed_dict={model.X: test_data})
        print(t1)
