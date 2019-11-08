import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from gan import GAN
from wgan import WGAN
# import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from mlxtend.data import loadlocal_mnist

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

tf.reset_default_graph()

def GANSampleZ(size):
    """
    Function:
        Randomly sample latent vector from uniform distribution.
    """
    return np.random.uniform(-1., 1., size=size)

def debug(model_name, dataset_name, num_hidden, latent_size, if_plot=True, if_save=False):
    """
    Function:
        For debugging.
    """
    if dataset_name == "MNIST":
        Mnist_local_path = "mnist/"
        X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path)
        X_train = X_train.reshape([-1, 28, 28, 1])
    elif dataset_name == "CIFAR":
        cifar_dir = "cifar-10-batches-py/"
        X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)
        X_train = TransferToImage(X_train)
    else:
        raise ValueError("Please input the right dataset name.")

    # parameters settings
    input_size = (32, 32, 3)
    batch_size = 256
    epochs = 10000
    hm_batches_train = len(X_train) // batch_size
    hidden_layer_size = 256 # feel free to tune

    if model_name == "VAE":
        aaaaa = 0
        # model = VAE(input_size, num_hidden, hidden_layer_size, latent_size)
        # generat_samples = model.Generating()
        # loss = model.Loss()
        # train_op = model.TrainModel()
    elif model_name == "GAN":
        model = GAN(input_size, num_hidden, hidden_layer_size, latent_size, batch_size)
        D_loss, G_loss, D_variables, G_variables = model.Loss()
        op_D, op_G = model.TrainModel(D_loss, G_loss, D_variables, G_variables)
        generat_samples = model.Generating()
    elif model_name == "WGAN":
        model = WGAN(input_size, num_hidden, hidden_layer_size, latent_size, batch_size)
        D_loss, G_loss, D_variables, G_variables = model.Loss()
        op_D, op_G = model.TrainModel(D_loss, G_loss, D_variables, G_variables)
        generat_samples = model.Generating()
    else:
        raise ValueError("Please input the right model name.")

    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)

    if if_plot:
        # live plot for visualizing learning
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        counter = 0
        step_counter = 0
        for epoch_id in tqdm(range(epochs)):
            np.random.shuffle(X_train)
            for each_batch_train in range(hm_batches_train):
                X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]

                if model_name == "VAE":
                    _, loss_vae = sess.run([train_op, loss], feed_dict={model.X: X_train_batch})
                    print([loss_vae])
                elif model_name == "GAN":
                    for whatever in range(5):
                        _, Dloss = sess.run([op_D, D_loss], feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    _, Gloss = sess.run([op_G, G_loss], feed_dict={model.Z: GANSampleZ([batch_size, latent_size])})
                    print([Dloss, Gloss])
                elif model_name == "WGAN":
                    for whatever in range(5):
                        _, Dloss, clip_d = sess.run([op_D, D_loss, clip_D], feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    _, Gloss = sess.run([op_G, G_loss], feed_dict={model.Z: GANSampleZ([batch_size, latent_size])})
                    print([Dloss, Gloss])

                if (epoch_id*batch_size + each_batch_train) % 50 == 0:
                    if model_name == "GAN" or model_name == "WGAN":
                        sample = sess.run(generat_samples, feed_dict={model.Z: GANSampleZ([batch_size, latent_size])})
                    elif model_name == "VAE":
                        sample = sess.run(generat_samples, feed_dict={model.X: X_test[:1], model.Z: VAESampleZ([batch_size, latent_size])})

                    if if_save:
                        dir_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size)
                        if not os.path.exists(dir_n):
                            os.mkdir(dir_n)
                        file_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size) + \
                                    "/" + str(counter) + ".npy"
                        np.save(file_n, sample)
                        counter += 1

                    current_samples = sample[:200]
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

                    if if_plot:
                        plt.cla()
                        ax.clear()
                        # ax.imshow(np.squeeze(sample))
                        ax.imshow(all_imgs)
                        fig.canvas.draw()
                        plt.pause(0.01)

            # t1 = sess.run(test, feed_dict={model.X: testdata, model.Z: GANSampleZ([batch_size, latent_size])})
            # print(t1.shape)




if __name__ == '__main__':
    # cifar_dir = "cifar-10-batches-py/"
    # X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)

    # X_train = TransferToImage(X_train)

    # gan = GAN()
    # gan.train(X_train, epochs=30000, batch_size=32, sample_interval=200)

    debug("GAN", "CIFAR", 1, 128)
