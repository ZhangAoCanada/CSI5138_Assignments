from gensim.models import KeyedVectors
import gensim
import tensorflow.compat.v1 as tf
import keras
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import matplotlib
import matplotlib.pyplot as plt
from keras.layers import Embedding, Dense, Flatten
from keras.layers import Input, LSTM, Dropout, SimpleRNN
from keras.models import Sequential, Model
from keras import optimizers

import os
import numpy as np
from pprint import pprint



def processReviews(paths):
    texts = []
    ratings = []

    for path in paths:
        for file in os.listdir(path):
            # get review
            rating = file.split('_')[1]
            rating = rating.split('.')[0]
            file = os.path.join(path, file)
            with open(file, "r", encoding='utf-8') as f:
                text = []
                for line in f:

                    # do some pre-processing and combine list of words for each review text
                    text += gensim.utils.simple_preprocess(line)
                texts.append(text)
                ratings.append(rating)
    # : 冒号
    return [texts, ratings]

Xtrain, ytrain = processReviews(["./aclImdb/train/neg/", "./aclImdb/train/pos/"])
Xtest, ytest = processReviews(["./aclImdb/test/neg/", "./aclImdb/test/pos/"])


texts = list(Xtrain + Xtest)
label = list(ytrain + ytest)

label = [int(a)>= 7 for a in label]



MAX_SEQUENCE_LENGTH=500

# tokenizer = tf.keras.preprocessing.text.Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(label))

print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
print(labels[0])

sequence_lens = sorted([len(s) for s in sequences])

# plots histogram
# plt.hist(sequence_lens, bins=100)
# plt.show()

embeddings_index = {}
glove_file = './glove.6B/glove.6B.50d.txt'

with open(glove_file, "r", encoding='utf-8') as f:
    for line in f:
        values = line.split()

        word = values[0]
        coefs = np.asarray(values[1:], dtype='float64')
        embeddings_index[word] = coefs

print('Found %s word vectors.' % len(embeddings_index))
EMBEDDING_DIM=50
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



X_train, y_train = data[:25000], labels[:25000]
X_test, y_test = data[25000:], labels[25000:]


# Choose RNN or NLTK with option
def RNN_or_NLTK(Option,state_size, MAX_SEQUENCE_LENGTH, learning_rate, Train_Text, Train_Label, Test_Text, Test_label):
    tf.disable_eager_execution()
    Text_1 = tf.placeholder(tf.int64, [None, MAX_SEQUENCE_LENGTH])
    Label_1 = tf.placeholder(tf.float64, [None, 2])
    initial_state = tf.nn.embedding_lookup(embedding_matrix, Text_1)
    Weight = tf.Variable(tf.random_normal_initializer()([state_size, 2]))
    Bias = tf.Variable(tf.random_normal_initializer()([2]))
    # Using tf.nn.rnn_cell MultiRNNCell to create three_layers RNN
    def get_a_cell():
        if Option == 1:
            Cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
        elif Option == 2:
            Cell = tf.nn.rnn_cell.BasicRNNCell(state_size)
        return Cell
    run_cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
    outputs, states = tf.nn.dynamic_rnn(run_cell, initial_state, dtype=tf.float64)
    Output_mean = tf.reduce_mean(outputs, axis=1)
    Weight = tf.cast(Weight, tf.float64)
    average_output = tf.cast(Output_mean, tf.float64)
    Bias = tf.cast(Bias, tf.float64)
    prediction = tf.matmul(average_output, Weight) + Bias
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Label_1, logits=prediction)
    loss = tf.reduce_mean(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    prediction = tf.nn.softmax(prediction)
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Label_1, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float64))
    session = tf.Session()
    with session as sess:
        sess.run(tf.global_variables_initializer())
        # Train network
        batch_size = 500
        epoches = 10
        dropout_prob = 0.4
        training_accuracy = []
        training_loss = []
        testing_accuracy = []
        testing_loss = []
        index_list = []
        index = np.arange(len(Train_Text))
        np.random.shuffle(index)
        X_train = Train_Text[index]
        y_train = Train_Label[index]
        iteraiton = len(Train_Text) // batch_size
        Loop_step = 0

        for epoch in range(epoches):
            print('epoch', epoch)

            for step in range(iteraiton):

                index_list.append(Loop_step)
                Loop_step +=1
                batch_x = X_train[step * batch_size: (step + 1) * batch_size]
                batch_y = y_train[step * batch_size: (step + 1) * batch_size]

                sess.run(optimizer, {Text_1: batch_x,
                              Label_1: batch_y})

                # Calculate batch loss and accuracy
                loss1 = sess.run(loss, feed_dict={Text_1: batch_x,
                                           Label_1: batch_y})

                acc = sess.run(accuracy, feed_dict={Text_1: batch_x,
                                              Label_1: batch_y})
                training_loss.append(loss1)
                training_accuracy.append(acc)

                test_loss = sess.run(loss, {Text_1: Test_Text[:256],
                                     Label_1: Test_label[:256]})
                test_acc = accuracy_score(Test_label.argmax(1),
                                      sess.run(prediction, {Text_1: Test_Text}).argmax(1))
                print("Step " + str(step) + ", Minibatch Loss= " + \
                  str(loss1) + ", Training Accuracy= " + \
                  str(acc) + ", Testing Accuracy:", \
                  str(test_acc))
                testing_accuracy.append(test_acc)
                testing_loss.append(test_loss)
    return training_accuracy, testing_accuracy, training_loss, testing_loss, index_list

train_acc, test_acc, train_loss, test_loss, index_list = RNN_or_NLTK(2, 50, MAX_SEQUENCE_LENGTH, 0.01, X_train, y_train, X_test, y_test)
plt.plot(index_list, train_acc, label='test')
plt.plot(index_list, test_acc, label='train')
plt.ylabel('accuracy')
plt.xlabel('steps')
plt.legend()
plt.show()

#
# def vanilla_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
#     model = Sequential()
#     model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False,
#                         weights=[embedding_matrix]))
#     model.add(SimpleRNN(units=state, input_shape=(num_words, 1), return_sequences=False))
#     model.add(Dropout(dropout))
#     model.add(Dense(num_outputs, activation='sigmoid'))
#     rmsprop = optimizers.RMSprop(lr=lra)
#     model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
#
#     return model
#
#
# def lstm_rnn(num_words, state, lra, dropout, num_outputs=2, emb_dim=50, input_length=500):
#     model = Sequential()
#     model.add(Embedding(input_dim=num_words + 1, output_dim=emb_dim, input_length=input_length, trainable=False,
#                         weights=[embedding_matrix]))
#     model.add(LSTM(state))
#     model.add(Dropout(dropout))
#     model.add(Dense(num_outputs, activation='sigmoid'))
#
#     rmsprop = optimizers.RMSprop(lr=lra)
#     model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
#
#     return model


# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = {'batch': [], 'epoch': []}
#         self.accuracy = {'batch': [], 'epoch': []}
#         self.val_loss = {'batch': [], 'epoch': []}
#         self.val_acc = {'batch': [], 'epoch': []}
#
#     def on_batch_end(self, batch, logs={}):
#         self.losses['batch'].append(logs.get('loss'))
#         self.accuracy['batch'].append(logs.get('acc'))
#         self.val_loss['batch'].append(logs.get('val_loss'))
#         self.val_acc['batch'].append(logs.get('val_acc'))
#
#     def on_epoch_end(self, batch, logs={}):
#         self.losses['epoch'].append(logs.get('loss'))
#         self.accuracy['epoch'].append(logs.get('acc'))
#         self.val_loss['epoch'].append(logs.get('val_loss'))
#         self.val_acc['epoch'].append(logs.get('val_acc'))
#
#     def loss_plot(self, loss_type):
#         iters = range(len(self.losses[loss_type]))
#         plt.figure()
#         # acc
#         plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
#         # loss
#         # plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
#         if loss_type == 'epoch':
#             # val_acc
#             plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
#             # val_loss
#             plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
#         plt.grid(True)
#         plt.xlabel(loss_type)
#         plt.ylabel('acc-loss')
#         plt.legend(loc="upper right")
#         plt.show()


# def runModel(state, lr, batch, dropout, model, epoch=5, num_outputs=2, emb_dim=100, input_length=2380):
#     num_words = len(word_index)
#     if model == "lstm":
#         model = lstm_rnn(num_words, state, lr, dropout)
#     elif model == "vanilla":
#         model = vanilla_rnn(num_words, state, lr, dropout)
#         epoch = 10
#
#     # model.summary()
#     # history = LossHistory()
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=batch, verbose=1)
#     # , callbacks = [history]
#     testscore = model.evaluate(X_test, y_test, verbose=0)
#     print('Test loss:', testscore[0])
#     print('Test accuracy:', testscore[1])
#     # history.loss_plot('batch')
#     return [history, testscore]
#
# batch_size = 128
# state = [20, 50, 100, 200, 500]
# learning_rate = 0.1
# dropout = 0.5
#
# for i in range(len(state)):
#     history = runModel(state[0], learning_rate, batch_size, dropout, 'vanilla')
#     state_parm = str('state' + state[0])
#     plt.plot(history.history['val_loss'], label = state_parm)
#
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()