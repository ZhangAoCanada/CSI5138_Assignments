##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
import numpy as np
import tensorflow as tf
from mlxtend.data import loadlocal_mnist
import pickle
from tqdm import tqdm

from VAE import VAE
from GAN_CNN import GAN
# from WGAN_CNN import WGAN

import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

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

############################## OTHER FUNCTIONS ############################
def VAESampleZ(size):
    """
    Function:
        Randomly sample latent vector from uniform distribution.
    """
    return np.random.randn(size[0], size[1])

def GANSampleZ(size):
    """
    Function:
        Randomly sample latent vector from uniform distribution.
    """
    return np.random.uniform(-1., 1., size=size)

def debug(model_name, dataset_name, num_hidden, latent_size, if_plot=False):
    """
    Function:
        For debugging.
    """
    if dataset_name == "MNIST":
        Mnist_local_path = "mnist/"
        X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path)
    elif dataset_name == "CIFAR":
        cifar_dir = "cifar-10-batches-py/"
        X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)
    else:
        raise ValueError("Please input the right dataset name.")

    # parameters settings
    batch_size = 256
    epochs = 1000
    hm_batches_train = len(X_train) // batch_size
    hidden_layer_size = 256 # feel free to tune

    testdata = X_train[:1].copy()

    if model_name == "VAE":
        model = VAE(input_size, num_hidden, hidden_layer_size, latent_size)
        generat_samples = model.Generating()
        loss = model.Loss()
        train_op = model.TrainModel()
    elif model_name == "GAN":
        model = GAN(input_size, num_hidden, hidden_layer_size, latent_size)
        generat_samples = model.Generating()
        loss_D, loss_G = model.Loss()
        train_op_D, train_op_G = model.TrainModel()
    elif model_name == "WGAN":
        model = WGAN(input_size, num_hidden, hidden_layer_size, latent_size)
        generat_samples = model.Generating()
        loss_D, loss_G = model.Loss()
        train_op_D, train_op_G = model.TrainModel()
        clip_D = model.ClipDiscriminatorWeights()
    else:
        raise ValueError("Please input the right model name.")
    
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)

    if if_plot:
        # live plot for visualizing learning
        fig = plt.figure()
        ax = fig.add_subplot(111)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        for epoch_id in tqdm(range(epochs)):
            np.random.shuffle(X_train)
            for each_batch_train in range(hm_batches_train):
                X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
                X_train_batch = TransferToImage(X_train_batch)

                if model_name == "VAE":
                    _, loss_vae = sess.run([train_op, loss], feed_dict={model.X: X_train_batch})
                    print([loss_vae])
                elif model_name == "GAN":
                    for whatever in range(5):
                        _, Dloss = sess.run([train_op_D, loss_D], feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    _, Gloss = sess.run([train_op_G, loss_G], feed_dict={model.Z: GANSampleZ([batch_size, latent_size])})
                    print([Dloss, Gloss])
                elif model_name == "WGAN":
                    for whatever in range(5):
                        _, Dloss, clip_d = sess.run([train_op_D, loss_D, clip_D], feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    _, Gloss = sess.run([train_op_G, loss_G], feed_dict={model.Z: GANSampleZ([batch_size, latent_size])})
                    print([Dloss, Gloss])

                if (epoch_id*batch_size + each_batch_train) % 20 == 0:
                    if model_name == "GAN" or model_name == "WGAN":
                        sample = sess.run(generat_samples, feed_dict={model.Z: GANSampleZ([1, latent_size])})
                    elif model_name == "VAE":
                        sample = sess.run(generat_samples, feed_dict={model.X: X_test[:1], model.Z: VAESampleZ([1, latent_size])})

                    if dataset_name == "MNIST":
                        sample = sample.reshape((-1, 28, 28))
                    # elif dataset_name == "CIFAR":
                    #     sample = TransferToImage(sample)

                    if if_plot:
                        plt.cla()
                        ax.clear()
                        # ax.imshow(np.concatenate([TransferToImage(testdata)[0], sample[0]], axis=1))
                        ax.imshow(sample[0])
                        fig.canvas.draw()
                        plt.pause(0.01)

def main(model_name, dataset_name, num_hidden, latent_size, hidden_layer_size, if_plot=False, if_save=True):
    """
    Function:
        Main Function, for training the model and testing
    """
    if dataset_name == "MNIST":
        Mnist_local_path = "mnist/"
        X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path)
    elif dataset_name == "CIFAR":
        cifar_dir = "cifar-10-batches-py/"
        X_train, Y_train, X_test, Y_test, input_size = LoadAllCifarData(cifar_dir)
    else:
        raise ValueError("Please input the right dataset name.")

    # parameters settings
    batch_size = 256
    epochs = 800
    sample_size = 30
    hm_batches_train = len(X_train) // batch_size    

    testdata = X_train[:1].copy()

    tf.reset_default_graph()

    if model_name == "VAE":
        model = VAE(input_size, num_hidden, hidden_layer_size, latent_size)
        generat_samples = model.Generating()
        loss = model.Loss()
        train_op = model.TrainModel()

        summary_loss_1 = tf.summary.scalar(dataset_name + "_" + "loss", loss)
    elif model_name == "GAN":
        model = GAN(input_size, num_hidden, hidden_layer_size, latent_size)
        generat_samples = model.Generating()
        loss_D, loss_G = model.Loss()
        train_op_D, train_op_G = model.TrainModel()

        summary_loss_1 = tf.summary.scalar(dataset_name + "_"  + "loss_D", loss_D)
        summary_loss_2 = tf.summary.scalar(dataset_name + "_"  + "loss_G", loss_G)
    elif model_name == "WGAN":
        model = WGAN(input_size, num_hidden, hidden_layer_size, latent_size)
        generat_samples = model.Generating()
        loss_D, loss_G = model.Loss()
        train_op_D, train_op_G = model.TrainModel()
        clip_D = model.ClipDiscriminatorWeights()

        summary_loss_1 = tf.summary.scalar(dataset_name + "_"  + "loss_D", loss_D)
        summary_loss_2 = tf.summary.scalar(dataset_name + "_"  + "loss_G", loss_G)
    else:
        raise ValueError("Please input the right model name.")

    log_dir = "logs/" + model_name + "_" + dataset_name + "_" + \
                str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size) + "/"
    
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(allow_growth=True)

    if if_plot:
        # live plot for visualizing learning
        fig = plt.figure()
        ax = fig.add_subplot(111)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summaries = log_dir
        train_writer = tf.summary.FileWriter(summaries, sess.graph)

        sess.run(init)
        counter = 0
        step_counter = 0
        for epoch_id in tqdm(range(epochs)):
            np.random.shuffle(X_train)
            for each_batch_train in range(hm_batches_train):
                X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]

                if model_name == "VAE":
                    _, loss_vae, sum_l, steps = sess.run([train_op, loss, summary_loss_1, model.global_step], 
                                                        feed_dict={model.X: X_train_batch})
                    train_writer.add_summary(sum_l, step_counter)
                    # print([loss_vae])
                elif model_name == "GAN":
                    for whatever in range(5):
                        _, Dloss = sess.run([train_op_D, loss_D], feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    _, Gloss, sum_l_d, sum_l_g, steps = sess.run([train_op_G, loss_G, summary_loss_1, summary_loss_2, model.global_step], \
                                        feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    train_writer.add_summary(sum_l_d, step_counter)                    
                    train_writer.add_summary(sum_l_g, step_counter)                    
                    # print([Dloss, Gloss])
                elif model_name == "WGAN":
                    for whatever in range(5):
                        _, Dloss, clip_d = sess.run([train_op_D, loss_D, clip_D], feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    _, Gloss, sum_l_d, sum_l_g, steps = sess.run([train_op_G, loss_G, summary_loss_1, summary_loss_2, model.global_step], \
                                        feed_dict={model.X: X_train_batch, model.Z: GANSampleZ([batch_size, latent_size])})
                    train_writer.add_summary(sum_l_d, step_counter)                    
                    train_writer.add_summary(sum_l_g, step_counter)  
                    # print([Dloss, Gloss])

                step_counter += 1

                if (epoch_id*batch_size + each_batch_train) % 500 == 0:
                    np.random.shuffle(X_test)
                    X_test_sample = X_test[:sample_size]
                    if model_name == "GAN" or model_name == "WGAN":
                        sample = sess.run(generat_samples, feed_dict={model.Z: GANSampleZ([sample_size, latent_size])})
                    elif model_name == "VAE":
                        sample = sess.run(generat_samples, feed_dict={model.X: X_test_sample, model.Z: VAESampleZ([sample_size, latent_size])})

                    if dataset_name == "MNIST":
                        sample = sample.reshape((-1, 28, 28))
                    elif dataset_name == "CIFAR":
                        sample = TransferToImage(sample)

                    if if_save:
                        dir_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size)
                        if not os.path.exists(dir_n):
                            os.mkdir(dir_n)
                        file_n = "samples/" + model_name + "_" + dataset_name + "_" + str(num_hidden) + "_" + str(latent_size) + "_" + str(hidden_layer_size) + \
                                    "/" + str(counter) + ".npy"
                        np.save(file_n, sample)
                        counter += 1

                    if if_plot:
                        plt.cla()
                        ax.clear()
                        # ax.imshow(np.concatenate([TransferToImage(testdata)[0], sample[0]], axis=1))
                        ax.imshow(sample[0])
                        fig.canvas.draw()
                        plt.pause(0.01)


if __name__ == "__main__":
    """
    model names:
        "VAE"
        "GAN"
        "WGAN"
    dataset names:
        "MNIST"
        "CIFAR"
    """
    model_names = ["VAE", "GAN", "WGAN"]
    dataset_names = ["CIFAR"]
    num_hidden = 0 # must <= 5
    latent_size = 512

    # num_hiddens = [2, 3]
    # latent_sizes = [128, 256, 512, 1024]
    # hidden_layer_sizes = [128, 256, 512, 1024]

    debug("GAN", "CIFAR", num_hidden, latent_size, True)
    # main(model_names[1], dataset_names[1], num_hiddens[2], latent_sizes[6], hidden_layer_sizes[1], False, True)

    # for hidden_layer_size in hidden_layer_sizes:

    # for dataset_name in dataset_names:
    #     for num_hidden in num_hiddens:
    #         for latent_size in latent_sizes:
    #             for model_name in model_names:
    #                 main(model_name, dataset_name, num_hidden, latent_size, 1024, False, True)


