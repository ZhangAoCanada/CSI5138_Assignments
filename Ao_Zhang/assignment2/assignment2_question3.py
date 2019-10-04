"""
CSI 5138: Assignment 2 ----- Question 3
Student:            Ao   Zhang
Student Number:     0300039680
"""
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#### other dependencies #####
import tensorflow as tf
import numpy as np
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm

class SoftmaxRegression:
    def __init__(self, input_size, output_size, dropout = False, BN = False):
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
        self.dropout_rate = 0.3

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

class MLP:
    def __init__(self, input_size, output_size, dropout = False, BN = False):
        self.input_size = input_size
        self.output_size = output_size
        self.X = tf.placeholder(tf.float32, shape = [None, self.input_size])
        self.Y = tf.placeholder(tf.float32, shape = [None, self.output_size])
        self.n_layer_1 = 256
        self.n_layer_2 = 256
        self.weights = {
                        'w1' : tf.Variable(tf.glorot_uniform_initializer()((self.input_size, self.n_layer_1))),
                        'w2' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_1, self.n_layer_2))),
                        'w3' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_2, self.output_size)))
                        }
        self.biases = {
                        'b1' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_1,))),
                        'b2' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_2,))),
                        'b3' : tf.Variable(tf.glorot_uniform_initializer()((self.output_size,)))
                        }
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 0.001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        100, 0.9, staircase=True)
        self.dropout = dropout
        self.BN = BN
        self.dropout_rate = 0.3

    def MultiLayers(self):
        layer_one = tf.nn.relu(tf.add(tf.matmul(self.X, self.weights['w1']), self.biases['b1']))
        if self.BN:
            layer_one = tf.layers.BatchNormalization()(layer_one)
        if self.dropout:
            layer_one = tf.nn.dropout(layer_one, self.dropout_rate)

        layer_two = tf.nn.relu(tf.add(tf.matmul(layer_one, self.weights['w2']), self.biases['b2']))
        if self.BN:
            layer_two = tf.layers.BatchNormalization()(layer_two)
        if self.dropout:
            layer_two = tf.nn.dropout(layer_two, self.dropout_rate)

        layer_three = tf.nn.relu(tf.add(tf.matmul(layer_two, self.weights['w3']), self.biases['b3']))
        if self.BN:
            layer_three = tf.layers.BatchNormalization()(layer_three)
        if self.dropout:
            layer_three = tf.nn.dropout(layer_three, self.dropout_rate)

        output = tf.nn.softmax(layer_three)
        return output

    def LossFunction(self):
        pred = self.MultiLayers()
        loss = tf.reduce_mean(tf.square(tf.add(pred, - self.Y)))
        return loss

    def TrainModel(self):
        loss = self.LossFunction()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        return learning_operation

    def Accuracy(self):
        pred = self.MultiLayers()
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
        return accuracy

# class CNN:


def TranslateLables(labels, num_class):
    hm_labels = len(labels)
    label_onehot = np.zeros((hm_labels, num_class), dtype = np.float32)
    for i in range(hm_labels):
        current_class = labels[i]
        label_onehot[i, current_class] = 1.
    return label_onehot

def GetMnistData(data_path, mode):
    X_train, Y_train_original = loadlocal_mnist(
            images_path=data_path + "train-images-idx3-ubyte", 
            labels_path=data_path + "train-labels-idx1-ubyte")

    X_test, Y_test_original = loadlocal_mnist(
            images_path=data_path + "t10k-images-idx3-ubyte", 
            labels_path=data_path + "t10k-labels-idx1-ubyte")

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    if mode == "CNN":
        X_train = X_train.reshape((-1, 28, 28))
        X_test = X_test.reshape((-1, 28, 28))

    all_classes = np.unique(Y_train_original)
    num_input = X_train.shape[1]
    num_class = len(all_classes)

    Y_train = TranslateLables(Y_train_original, num_class)
    Y_test = TranslateLables(Y_test_original, num_class)

    return X_train, Y_train, X_test, Y_test, num_input, num_class

def FolderName(mode, dropout, BN):
    original_name = '/' + mode
    if dropout:
        original_name += '_dropout'
    if BN:
        original_name += '_BN'
    return original_name

if __name__ == "__main__":
    # basical settings
    epoches = 50
    batch_size = 500
    mode = "MLP"
    dropout = False
    BN = True
    Mnist_local_path = "mnist/"

    # read Mnist
    X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path, mode)

    # training parameter
    hm_batches_train = len(X_train) // batch_size

    if mode == "softmax":
        model = SoftmaxRegression(input_size, output_size, dropout, BN)
    elif mode == "MLP":
        model = MLP(input_size, output_size, dropout, BN)
    # elif mode == "CNN":

    else:
        raise ValueError("Wrong Mode Input, please doublecheck it.")

    loss = model.LossFunction()
    accuracy = model.Accuracy()
    
    loss_graph_name = "loss"
    acc_graph_name = "accuracy"
    summary_loss = tf.summary.scalar(loss_graph_name, loss)
    summary_accuracy = tf.summary.scalar(acc_graph_name, accuracy)

    train = model.TrainModel()

    # initialization
    init = tf.global_variables_initializer()

    # GPU settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summaries_train = 'logs/train/'
        summaries_test = 'logs/test/'
        folder_name = FolderName(mode, dropout, BN)
        train_writer = tf.summary.FileWriter(summaries_train + folder_name, sess.graph)
        test_writer = tf.summary.FileWriter(summaries_test + folder_name, sess.graph)
        sess.run(init)
        for each_epoch in tqdm(range(epoches)):
            for each_batch_train in range(hm_batches_train):
                X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
                Y_train_batch = Y_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]

                _, loss_val, summary_l, steps = sess.run([train, loss, summary_loss, model.global_step], \
                                                                    feed_dict = {model.X : X_train_batch, \
                                                                                model.Y : Y_train_batch})

                train_writer.add_summary(summary_l, steps)

                summary_a = sess.run(summary_accuracy, feed_dict = {model.X : X_test, \
                                                                    model.Y : Y_test})

                test_writer.add_summary(summary_a, steps)

        
