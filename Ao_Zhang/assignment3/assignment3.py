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
# import matplotlib
# matplotlib.use("tkagg")
# import matplotlib.pyplot as plt
import os
import re
import numpy as np
from glob import glob
import tensorflow as tf
from tqdm import tqdm

class WordVectorAndList:
    def __init__(self, vocab_file, vector_file):
        self.vocab_file = vocab_file
        self.vector_file = vector_file
        self.vocab_list = self.VocabList(self.vocab_file)
        if os.path.exists("wordvector.npy"):
            self.word_vector = np.load("wordvector.npy")
        else:
            self.word_vector = self.WordVector(self.vector_file, self.vocab_list)

    def VocabList(self, filename):
        with open(filename, "r") as f:
            all_lines = f.readlines()
            vocab_dict = {}
            count = 0
            for line in all_lines:
                count += 1
                words = line.split()
                vocab_word = words[0]
                vocab_dict[vocab_word] = count
        return vocab_dict

    def WordVector(self, filename, vocab_list):
        with open(filename, "r") as f:
            all_lines = f.readlines()
            output_array = np.zeros((len(vocab_list) + 1, 50), dtype = np.float32)
            for line in all_lines:
                characters = line.split()
                if characters[0] not in vocab_list:
                    continue
                word_ind = vocab_list[characters[0]]
                word_vector = []
                for i in range(1, len(characters)):
                    word_vector.append(float(characters[i]))
                word_vector = np.array(word_vector)
                output_array[word_ind] = word_vector
        np.save("wordvector.npy", output_array)
        return output_array

#################################################################################
def FindAllSequanceLen(dataset_names, all_length):
    for file_name in dataset_names:
        with open(file_name, "r") as f:
            line = f.readline()
            words = line.split()
            all_length.append(len(words))
    return all_length

def PlotLenHist(train_pos_files, train_neg_files, test_pos_files, test_neg_files):
    all_length = []
    all_length = FindAllSequanceLen(train_pos_files, all_length)
    all_length = FindAllSequanceLen(train_neg_files, all_length)
    all_length = FindAllSequanceLen(test_pos_files, all_length)
    all_length = FindAllSequanceLen(test_neg_files, all_length)
    plt.hist(all_length, bins = 500)
    plt.title("Histogram of sequance length")
    plt.xlabel("length bins")
    plt.ylabel("number of sequences")
    plt.show()

#################################################################################
def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def ReadDatasetAsWordIndex(all_filenames, word_list, len_lim):
    all_dataset = []
    for file_name in all_filenames:
        output_dataset = np.zeros((len_lim), dtype = np.int32)
        f = open(file_name, "r")
        line=f.readline()
        cleaned_line = cleanSentences(line)
        words = cleaned_line.split()
        for i in range(len(words)):
            if i < len_lim:
                output_dataset[i] = word_list[words[i]]
            else:
                continue
        all_dataset.append(output_dataset)
    all_dataset = np.array(all_dataset)
    return all_dataset

def GetAllData(train_pos_files, train_neg_files, test_pos_files, test_neg_files, length_limit):
    train_pos_set = ReadDatasetAsWordIndex(train_pos_files, word_list, length_limit)
    train_neg_set = ReadDatasetAsWordIndex(train_neg_files, word_list, length_limit)
    test_pos_set = ReadDatasetAsWordIndex(test_pos_files, word_list, length_limit)
    test_neg_set = ReadDatasetAsWordIndex(test_neg_files, word_list, length_limit)

    return train_pos_set, train_neg_set, test_pos_set, test_neg_set

def CreatDataSet(pos_data, neg_data, name_prefix):
    len_pos_data = len(pos_data)
    len_neg_data = len(neg_data)
    pos_label = np.zeros((len_pos_data, 2), dtype = np.float32)
    pos_label[:, 0] = 1.
    neg_label = np.zeros((len_neg_data, 2), dtype = np.float32)
    neg_label[:, 1] = 1.
    all_dataset = np.concatenate([pos_data, neg_data], axis = 0)
    all_labels = np.concatenate([pos_label, neg_label], axis = 0)
    assert len(all_dataset) == len(all_labels)
    indexes = np.arange(len(all_dataset))
    np.random.shuffle(indexes)
    dataset = all_dataset[indexes]
    labels = all_labels[indexes]
    np.save(name_prefix + "_dataset.npy", dataset)
    np.save(name_prefix + "_labels.npy", labels)
    return dataset, labels

def GetTrainAndTestSets(train_pos_files, train_neg_files, test_pos_files, 
                        test_neg_files, length_limit):
    existance = os.path.exists("training_dataset.npy") and os.path.exists("training_labels.npy") \
                and os.path.exists("test_dataset.npy") and os.path.exists("test_labels.npy")
    if not existance:
        train_pos_set, train_neg_set, test_pos_set, test_neg_set = GetAllData(train_pos_files, \
                                    train_neg_files, test_pos_files, test_neg_files, length_limit)
        training_set, training_label = CreatDataSet(train_pos_set, train_neg_set, name_prefix = "training")
        test_set, test_label = CreatDataSet(test_pos_set, test_neg_set, name_prefix = "test")
    else:
        training_set = np.load("training_dataset.npy")
        training_label = np.load("training_labels.npy")
        test_set = np.load("test_dataset.npy")
        test_label = np.load("test_labels.npy")
    return training_set, training_label, test_set, test_label

#################################################################################
class AssignmentRNNModel:
    def __init__(self, length_limit, word_vector, state_size, name):
        self.name = name
        self.word_vector = word_vector
        self.rnn_size = state_size
        self.length_limit = length_limit
        self.learning_rate_start = 0.001
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        100, 0.96, staircase=True)
        self.X_ids = tf.placeholder(tf.int32, [None, self.length_limit])
        self.Y = tf.placeholder(tf.float32, [None, 2])
        self.X = tf.nn.embedding_lookup(self.word_vector, self.X_ids)
        self.W_out = tf.Variable(tf.random_normal_initializer()([self.rnn_size, 2]))
        self.B_out = tf.Variable(tf.random_normal_initializer()([2]))
        if self.name == "vanilla":
            self.rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.rnn_size)
        elif self.name == "lstm":
            self.rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
        else:
            raise ValueError("wrong input name, please choose one from either vanilla or lstm")
    
    def RnnModel(self):
        outputs, states = tf.nn.dynamic_rnn(self.rnn_cell, self.X, dtype=tf.float32)

        last_output = tf.reduce_mean(outputs, axis = 1)
        prediction = tf.matmul(last_output, self.W_out) + self.B_out
        return prediction
    
    def LossFunction(self):
        """
        Function:
            Define loss function as cross entropy after softmax gating function.
        """
        pred = self.RnnModel()
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
        pred = self.RnnModel()
        pred = tf.nn.softmax(pred)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype = tf.float32))
        return accuracy


#################################################################################
def main(mode, state_size):
    vocab_file = "vocab.txt"
    vector_file = "vectors.txt"

    train_pos_files = glob("aclImdb/train/pos/*.txt")
    train_neg_files = glob("aclImdb/train/neg/*.txt")

    test_pos_files = glob("aclImdb/test/pos/*.txt")
    test_neg_files = glob("aclImdb/test/neg/*.txt")

    plot_hist = False
    length_limit = 500

    batch_size = 500
    epoches = 100
    # state_size_requirements = [20, 50, 100, 200, 500]
    # state_size = state_size_requirements[1]
    # mode = "vanilla"

    words_dict_creater = WordVectorAndList(vocab_file, vector_file)
    word_list = words_dict_creater.vocab_list
    word_vector = words_dict_creater.word_vector

    if plot_hist:
        PlotLenHist(train_pos_files, train_neg_files, test_pos_files, test_neg_files)

    training_set, training_label, test_set, test_label = GetTrainAndTestSets(train_pos_files, \
                                train_neg_files, test_pos_files, test_neg_files, length_limit)

    num_train_batch = len(training_set) // batch_size
    num_test_batch = len(test_set) // batch_size
    print([num_train_batch, num_test_batch])

    #################################################################################
    # tf.reset_default_graph()

    model = AssignmentRNNModel(length_limit, word_vector, state_size, name = mode)

    # if mode == "vanilla":
    #     model = AssignmentRNNModel(length_limit, word_vector, state_size, name = "vanilla")
    # elif mode == "lstm":
    #     model = AssignmentRNNModel(length_limit, word_vector, state_size, name = "lstm")
    # else:
    #     raise ValueError("wrong input name, please choose one from either vanilla or lstm")

    loss = model.LossFunction()
    accuracy = model.Accuracy()
    train = model.TrainModel()

    # tensorboard settings
    loss_graph_name = "loss"
    acc_graph_name = "accuracy"
    summary_loss = tf.summary.scalar(loss_graph_name, loss)
    streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(accuracy)
    summary_accuracy = tf.summary.scalar(acc_graph_name, streaming_accuracy)

    # initialization
    init = tf.global_variables_initializer()
    # GPU settings
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        summaries_train = 'logs/train/'
        summaries_test = 'logs/test/'
        folder_name = mode + "_" + str(state_size)
        train_writer = tf.summary.FileWriter(summaries_train + folder_name, sess.graph)
        test_writer = tf.summary.FileWriter(summaries_test + folder_name, sess.graph)
        # summary_acc = tf.Summary()
        sess.run(init)
        for epoch_id in tqdm(range(epoches)):
            for train_batch_id in range(num_train_batch):
                X_train_batch = training_set[train_batch_id*batch_size : (train_batch_id+1)*batch_size]
                Y_train_batch = training_label[train_batch_id*batch_size : (train_batch_id+1)*batch_size]
                _, loss_val, summary_l, steps = sess.run([train, loss, summary_loss, model.global_step], \
                                                                    feed_dict = {model.X_ids : X_train_batch, \
                                                                                model.Y : Y_train_batch})
                train_writer.add_summary(summary_l, steps)

                """
                When GPU memory is not enough
                """
                if train_batch_id % 10 == 0:
                    sess.run(tf.local_variables_initializer())
                    for test_batch_id in range(num_test_batch):
                        X_test_batch = test_set[test_batch_id*batch_size: (test_batch_id+1)*batch_size]
                        Y_test_batch = test_label[test_batch_id*batch_size: (test_batch_id+1)*batch_size]
                        sess.run([streaming_accuracy_update], feed_dict = {model.X_ids : X_test_batch, \
                                                                            model.Y : Y_test_batch})

                    summary_a = sess.run(summary_accuracy)
                    test_writer.add_summary(summary_a, steps)

if __name__ == "__main__":
    """
    2 modes: "vanilla", "lstm"
    """
    mode = "vanilla"
    state_size_requirements = [20, 50, 100, 200, 500]
    state_size = state_size_requirements[1]
    main(mode, state_size)




