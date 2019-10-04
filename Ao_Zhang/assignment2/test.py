"""
CSI 5138: Assignment 2 ----- Question 3
Student:            Ao   Zhang
Student Number:     0300039680
"""
#### for plotting through X11 #####
# import matplotlib
# matplotlib.use("tkagg")
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
# import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm

class SoftmaxRegression:
    def __init__(self, input_size, output_size, dropout = False, BN = True):
        self.input_size = input_size
        self.output_size = output_size
        self.X = tf.placeholder(tf.float32, shape = [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, shape = [None, self.output_size])
        self.weights = tf.Variable(tf.glorot_uniform_initializer()((self.input_size, self.output_size)))
        self.biases = tf.Variable(tf.glorot_uniform_initializer()((self.output_size,)))
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 0.001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        100, 0.9, staircase=True)
        self.dropout = dropout
        self.BN = BN
        self.dropout_rate = 0.5

    def Regression(self):
        layer = tf.add(tf.matmul(self.X, self.weights), self.biases)
        if self.BN:
            layer = tf.layers.BatchNormalization()(layer)
        if self.dropout:
            layer = tf.nn.dropout(layer, self.dropout_rate)
        output = tf.nn.softmax(layer)
        return output

    def LossFunction(self):
        pred = self.Regression()
        loss = tf.reduce_mean(tf.square(tf.add(pred, - self.Y)))
        return loss

    def TrainModel(self):
        loss = self.LossFunction()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        return learning_operation

    def Accuracy(self):
        pred = self.Regression()
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
        return accuracy

def TranslateLables(labels, num_class):
    hm_labels = len(labels)
    label_onehot = np.zeros((hm_labels, num_class), dtype = np.float32)
    for i in range(hm_labels):
        current_class = labels[i]
        label_onehot[i, current_class] = 1.
    return label_onehot

def GetMnistData(data_path):
    X_train, Y_train_original = loadlocal_mnist(
            images_path=data_path + "train-images-idx3-ubyte", 
            labels_path=data_path + "train-labels-idx1-ubyte")

    X_test, Y_test_original = loadlocal_mnist(
            images_path=data_path + "t10k-images-idx3-ubyte", 
            labels_path=data_path + "t10k-labels-idx1-ubyte")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    all_classes = np.unique(Y_train_original)
    num_input = X_train.shape[1]
    num_class = len(all_classes)

    Y_train = TranslateLables(Y_train_original, num_class)
    Y_test = TranslateLables(Y_test_original, num_class)

    return X_train, Y_train, X_test, Y_test, num_input, num_class

if __name__ == "__main__":
    # read Mnist
    Mnist_local_path = "mnist/"
    X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path)
    
    # basical settings
    epoches = 50
    batch_size = 500
    hm_batches_train = len(X_train) // batch_size
    hm_batches_test = len(X_test) // batch_size

    # model
    softmax_regression = SoftmaxRegression(input_size, output_size)
    loss = softmax_regression.LossFunction()
    accuracy = softmax_regression.Accuracy()

    # summary_loss = tf.summary.scalar("loss", loss)
    # summary_accuracy = tf.summary.scalar("accuracy", accuracy)

    train = softmax_regression.TrainModel()

    # merged = tf.summary.merge_all()

    # initialization
    init = tf.global_variables_initializer()

    # GPU settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # summaries_train = 'logs/train/'
        # summaries_test = 'logs/test/'
        # train_writer = tf.summary.FileWriter(summaries_train + '/softmax', sess.graph)
        # test_writer = tf.summary.FileWriter(summaries_test + '/softmax', sess.graph)
        sess.run(init)
        for each_epoch in tqdm(range(epoches)):
            for each_batch_train in range(hm_batches_train):
                X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
                Y_train_batch = Y_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]

                _, loss_val, steps = sess.run([train, loss, softmax_regression.global_step], \
                                                                    feed_dict = {softmax_regression.X : X_train_batch, \
                                                                                softmax_regression.Y : Y_train_batch})
                
                print(loss_val)
                # steps = each_epoch * hm_batches + each_batch_train
                # train_writer.add_summary(summary_l, steps)

                # current_acc = 0.

                # for each_batch_test in range(hm_batches_test):
                # X_test_batch = X_test[each_batch_test*batch_size: (each_batch_test+1)*batch_size]
                # Y_test_batch = Y_test[each_batch_test*batch_size: (each_batch_test+1)*batch_size]

                # summary_a = sess.run(summary_accuracy, feed_dict = {softmax_regression.X : X_test, \
                #                                                     softmax_regression.Y : Y_test})

                # current_acc += summary_a
                # current_acc /= hm_batches_test

                # test_writer.add_summary(summary_a, steps)




    # plt.ion()
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # for i in range(len(X)):
    #     image = X[i].reshape(28, 28)

    #     plt.cla()
    #     ax1.clear()
    #     ax1.imshow(image)
    #     fig.canvas.draw()
        
