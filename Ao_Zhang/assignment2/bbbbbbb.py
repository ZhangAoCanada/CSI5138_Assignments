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
import tensorflow.keras.layers as L
import numpy as np
from mlxtend.data import loadlocal_mnist
from tqdm import tqdm

class CNN():
    def __init__(self, input_size, output_size, dropout = False, BN = False):
        """
        Function:
            Initialization of all variables
        """
        self.w, self.h = input_size
        self.output_size = output_size
        self.X = tf.placeholder(tf.float32, shape = [None, self.w, self.h, 1])
        self.Y = tf.placeholder(tf.float32, shape = [None, self.output_size])
        self.input_channel = 1
        self.n_feature_1 = 32
        self.n_feature_2 = 64
        self.n_feature_3 = 64
        self.n_layers_1 = 3 * 3 * 64
        self.weights = {
                        'w6' : tf.Variable(tf.glorot_uniform_initializer()([self.n_layers_1, self.output_size]))
                        }
        self.biases = {
                        'b6' : tf.Variable(tf.glorot_uniform_initializer()((self.output_size,)))
                        }
        # define convolutional kernels and maxpool kernels

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 0.001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        100, 0.96, staircase=True)
        self.dropout = dropout
        self.BN = BN
        self.dropout_rate = 0.5

    def ConvolutionalLayer(self, input_data, filters, kernel, strides, padding):
        """
        Function:
            Define convolutional layer.
        """
        layer = L.Conv2D(filters, kernel, strides = strides, padding = padding)(input_data)
        if self.BN:
            layer =  L.BatchNormalization()(layer)
        layer = L.ReLU()(layer)
        return layer

    def MaxPool(self, input_data):
        """
        Function:
            Define maxpool layer.
        """
        layer = L.MaxPooling2D()(input_data)
        return layer

    def LastLayer(self, input_data, weights, biases):
        """
        Function:
            Define last layer.
        """
        input_data = tf.reshape(input_data, [tf.shape(input_data)[0], -1])
        layer = tf.add(tf.matmul(input_data, weights), biases)
        return layer

    def ConvolutionNet(self):
        """
        Function:
            Define Convolutional Nets as:
            (batch, 28, 28, 1) -> (batch, 26, 26, 32) -> (batch, 13, 13, 32) -> 
            (batch, 11, 11, 64) -> (batch, 6, 6, 64) -> (batch, 4, 4, 64) ->
            (batch, 4 * 4 * 64) -> (batch, 10)
        """
        layer = self.ConvolutionalLayer(self.X, 32, [3,3], [1,1], "valid")
        layer = self.MaxPool(layer)
        layer = self.ConvolutionalLayer(layer, 64, [3,3], [1,1], "valid")
        layer = self.MaxPool(layer)
        layer = self.ConvolutionalLayer(layer, 64, [3,3], [1,1], "valid")

        layer = self.LastLayer(layer, self.weights['w6'], self.biases['b6'])
        return layer

    def LossFunction(self):
        """
        Function:
            Define loss function as cross entropy after softmax gating function.
        """
        pred = self.ConvolutionNet()
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.Y, logits = pred)
        loss = tf.reduce_mean(loss)
        return loss

    def TrainModel(self):
        """
        Function:
            Define optimization method as tf.train.AdamOptimizer()
        """
        loss = self.LossFunction()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        return learning_operation

    def Accuracy(self):
        """
        Function:
            Define validation as comparing the prediction with groundtruth. Details would
        be, finding the predction class w.r.t the largest probability, then compare with 
        the true labels.
        """
        pred = self.ConvolutionNet()
        pred = tf.nn.softmax(pred)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
        return accuracy

#################################################################################
################################ Data Processing ################################
#################################################################################
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

def GetMnistData(data_path, mode):
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

    # if CNN, reshape data into image format
    if mode == "CNN":
        X_train = np.expand_dims(X_train.reshape((-1, 28, 28)), axis = -1)
        X_test = np.expand_dims(X_test.reshape((-1, 28, 28)), axis = -1)
        
    # find how many classes
    all_classes = np.unique(Y_train_original)
    num_class = len(all_classes)
    
    # if CNN, change the input shape into image shape
    if mode == "CNN":
        num_input = (28, 28)
    else:
        num_input = X_train.shape[1]    

    # transfer label format
    Y_train = TranslateLables(Y_train_original, num_class)
    Y_test = TranslateLables(Y_test_original, num_class)
    return X_train, Y_train, X_test, Y_test, num_input, num_class

def FolderName(mode, dropout, BN):
    """
    Function:
        For creating differnt folder for tf.summary() in order to monitor
    the training process.
    """
    original_name = '/' + mode
    if dropout:
        original_name += '_dropout'
    if BN:
        original_name += '_BN'
    return original_name

#################################################################################
#################################### Running ####################################
#################################################################################
if __name__ == "__main__":
    """
    3 modes:
        "softmax"
        "MLP"  
        "CNN"
    """

    mode = "CNN"
    dropout = False
    BN = False

    # basical settings
    epoches = 20
    batch_size = 500
    Mnist_local_path = "mnist/"

    # read Mnist
    X_train, Y_train, X_test, Y_test, input_size, output_size = GetMnistData(Mnist_local_path, mode)

    # training parameter
    hm_batches_train = len(X_train) // batch_size
    hm_batches_test = len(X_test) // batch_size

    if mode == "softmax":
        model = SoftmaxRegression(input_size, output_size, dropout, BN)
    elif mode == "MLP":
        model = MLP(input_size, output_size, dropout, BN)
    elif mode == "CNN":
        model = CNN(input_size, output_size, dropout, BN)
    else:
        raise ValueError("Wrong Mode Input, please doublecheck it.")

    loss = model.LossFunction()
    # loss = tf.reduce_mean(loss)
    accuracy = model.Accuracy()
    
    # loss_graph_name = "loss"
    # acc_graph_name = "accuracy"
    summary_loss = tf.summary.scalar("111", loss)
    # streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(accuracy)
    # summary_accuracy = tf.summary.scalar(acc_graph_name, streaming_accuracy)
    # summary_accuracy_straight = tf.summary.scalar(acc_graph_name, accuracy)

    train = model.TrainModel()

    # initialization
    init = tf.global_variables_initializer()

    # GPU settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # summaries_train = 'logs/train/'
        # summaries_test = 'logs/test/'
        # folder_name = FolderName(mode, dropout, BN)
        # train_writer = tf.summary.FileWriter(summaries_train + folder_name, sess.graph)
        # test_writer = tf.summary.FileWriter(summaries_test + folder_name, sess.graph)
        # summary_acc = tf.Summary()
        sess.run(init)
        for each_epoch in tqdm(range(epoches)):
            for each_batch_train in range(hm_batches_train):
                X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
                Y_train_batch = Y_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]

                _, loss_val, summary_l, steps = sess.run([train, loss, summary_loss, model.global_step], \
                                                                    feed_dict = {model.X : X_train_batch, \
                                                                                model.Y : Y_train_batch})

                print(loss_val)
                # train_writer.add_summary(summary_l, steps)
                
                # """
                # When GPU memory is not enough
                # """
                # sess.run(tf.local_variables_initializer())
                # for each_batch_test in range(hm_batches_test):
                #     X_test_batch = X_test[each_batch_test*batch_size: (each_batch_test+1)*batch_size]
                #     Y_test_batch = Y_test[each_batch_test*batch_size: (each_batch_test+1)*batch_size]
                #     sess.run([streaming_accuracy_update], feed_dict = {model.X : X_test_batch, \
                #                                             model.Y : Y_test_batch})

                # summary_a = sess.run(summary_accuracy)
                # test_writer.add_summary(summary_a, steps)




