import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import os
import keras
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.metrics import roc_auc_score, accuracy_score
tf.disable_v2_behavior()

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'

train_images_idx3_ubyte_file = 'data/train-images-idx3-ubyte'
train_labels_idx1_ubyte_file = 'data/train-labels-idx1-ubyte'
test_images_idx3_ubyte_file = 'data/t10k-images-idx3-ubyte'
test_labels_idx1_ubyte_file = 'data/t10k-labels-idx1-ubyte'


def decode_idx3_ubyte(idx3_ubyte_file):
    bin_data = open(idx3_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('number of magic number:%d, number of pics: %d, size of pics: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    bin_data = open(idx1_ubyte_file, 'rb').read()
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('number of magic number:%d, number of pics: %d' % (magic_number, num_images))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels

def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    """
    return decode_idx3_ubyte(idx_ubyte_file)

def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    """
    return decode_idx1_ubyte(idx_ubyte_file)

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).


    """
    return decode_idx3_ubyte(idx_ubyte_file)

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.


    """
    return decode_idx1_ubyte(idx_ubyte_file)


def load_data():
    train_images1 = load_train_images()
    train_images = train_images1.reshape(60000, 784)
    train_labels = load_train_labels()
    Train_labels_CNN = train_labels
    test_images = load_test_images()
    test_images2 = test_images.reshape(10000, 784)
    test_labels = load_test_labels()
    Test_labels_CNN = test_labels

    train_images = train_images.astype(np.float32)
    test_images = train_labels.astype(np.float32)

    train_labels = Translate_y(train_labels, 10)
    test_labels = Translate_y(test_labels, 10)

    # #test1
    # for i in range(10):
    #     print(train_labels[i])
    #     plt.imshow(train_images[i], cmap='gray')
    #     plt.show()
    # print ('done')

    #test2
    # plt.figure(1)
    # fig, ax = plt.subplots(10, 10)
    # k = 0
    # for i in range(10):
    #     for j in range(10):
    #         ax[i][j].imshow(train_images1[k], aspect='auto')
    #         k += 1
    #
    # plt.show()
    return train_images, train_labels, test_images2, test_labels, Train_labels_CNN, Test_labels_CNN

def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu

def Translate_y(labels, num_class):
    hm_labels = len(labels)
    label_onehot = np.zeros((hm_labels, num_class), dtype = np.float32)
    for i in range(hm_labels):
        current_class = labels[i]
        current_class = int(current_class)
        label_onehot[i, current_class] = 1.
    return label_onehot

def MLP(Train_image, Train_label, Test_images, Test_labels):
    s = tf.InteractiveSession()
    ## Defining various initialization parameters for 784-512-256-10 MLP model
    num_classes = Test_labels.shape[1]
    num_features = Train_image.shape[1]
    num_output = Train_label.shape[1]
    num_layers_0 = 512
    num_layers_1 = 256
    starter_learning_rate = 0.001
    regularizer_rate = 0.1
    # Placeholders for the input data
    input_X = tf.placeholder('float32', shape=(None, num_features), name="input_X")
    input_y = tf.placeholder('float32', shape=(None, num_classes), name='input_Y')
    ## for dropout layer
    keep_prob = tf.placeholder(tf.float32)
    ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
    weights_0 = tf.Variable(tf.random_normal([num_features, num_layers_0], stddev=(1 / tf.sqrt(float(num_features)))))
    bias_0 = tf.Variable(tf.random_normal([num_layers_0]))
    weights_1 = tf.Variable(tf.random_normal([num_layers_0, num_layers_1], stddev=(1 / tf.sqrt(float(num_layers_0)))))
    bias_1 = tf.Variable(tf.random_normal([num_layers_1]))
    weights_2 = tf.Variable(tf.random_normal([num_layers_1, num_output], stddev=(1 / tf.sqrt(float(num_layers_1)))))
    bias_2 = tf.Variable(tf.random_normal([num_output]))
    ## Initializing weigths and biases
    hidden_output_0 = tf.nn.relu(tf.matmul(input_X, weights_0) + bias_0)
    hidden_output_0_0 = tf.nn.dropout(hidden_output_0, keep_prob)
    hidden_output_1 = tf.nn.relu(tf.matmul(hidden_output_0_0, weights_1) + bias_1)
    hidden_output_1_1 = tf.nn.dropout(hidden_output_1, keep_prob)
    predicted_y = tf.sigmoid(tf.matmul(hidden_output_1_1, weights_2) + bias_2)
    ## Defining the loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predicted_y, labels=input_y)) \
           + regularizer_rate * (tf.reduce_sum(tf.square(bias_0)) + tf.reduce_sum(tf.square(bias_1)))
    ## Variable learning rate
    learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
    ## Adam optimzer for finding the right weight
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=[weights_0, weights_1, weights_2,
                                                                               bias_0, bias_1, bias_2])
    ## Metrics definition
    correct_prediction = tf.equal(tf.argmax(Train_label, 1), tf.argmax(predicted_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    ## Training parameters
    batch_size = 128
    epochs = 14
    dropout_prob = 0.6
    training_accuracy = []
    training_loss = []
    testing_accuracy = []
    s.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        arr = np.arange(Train_image.shape[0])
        np.random.shuffle(arr)
        for index in range(0, Train_image.shape[0], batch_size):
            s.run(optimizer, {input_X: Train_image[arr[index:index + batch_size]],
                              input_y: Train_label[arr[index:index + batch_size]],
                              keep_prob: dropout_prob})
        training_accuracy.append(s.run(accuracy, feed_dict={input_X: Train_image,
                                                            input_y: Train_label, keep_prob: 1}))
        training_loss.append(s.run(loss, {input_X: Train_image,
                                          input_y: Train_label, keep_prob: 1}))

        ## Evaluation of model
        testing_accuracy.append(accuracy_score(Test_labels.argmax(1),
                                               s.run(predicted_y, {input_X: Test_images, keep_prob: 1}).argmax(1)))
        print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}, Test acc:{3:.3f}".format(epoch,
                                                                                           training_loss[epoch],
                                                                                           training_accuracy[epoch],

                                                                                    testing_accuracy[epoch]))

    iterations = list(range(epochs))
    # plt.figure(2)
    plt.plot(iterations, training_accuracy, label='Train')
    plt.plot(iterations, testing_accuracy, label='Test')
    plt.ylabel('Accuracy')
    plt.xlabel('iterations')
    plt.legend()

    plt.show()
    print("Train Accuracy: {0:.2f}".format(training_accuracy[-1]))
    print("Test Accuracy:{0:.2f}".format(testing_accuracy[-1]))



def Softmax(train_images, train_labels, test_images, test_labels):
    num_features = 784
    # number of target labels
    num_labels = 10
    # learning rate (alpha)
    learning_rate = 0.05
    # batch size
    batch_size = 128
    # number of epochs
    num_steps = 5001

    # input data
    train_dataset = train_images
    train_labels = train_labels
    test_dataset = test_images
    test_labels = test_labels

    # initialize a tensorflow graph
    graph = tf.Graph()

    with graph.as_default():
        tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_test_dataset = tf.constant(test_dataset)

        # Variables.
        weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
        biases = tf.Variable(tf.zeros([num_labels]))

        # Training computation.
        logits = tf.matmul(tf_train_dataset, weights) + biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf_train_labels, logits=logits))

        # Optimizer.
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

        # Predictions for the training, validation, and test data.
        train_prediction = tf.nn.softmax(logits)
        print(np.shape(tf_test_dataset))
        print(np.shape(weights))
        print(np.shape(biases))
        # multul = tf.matmul(tf_test_dataset, weights).astype(np.float32)
        test_prediction = tf.nn.softmax(tf.matmul(tf.cast(tf_test_dataset, tf.float32), tf.cast(weights, tf.float32)) + biases)
        with tf.Session(graph=graph) as session:
            # initialize weights and biases
            tf.global_variables_initializer().run()
            print("Initialized")

            for step in range(num_steps):
                # pick a randomized offset
                offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)
                # Generate a minibatch.
                batch_data = train_dataset[offset:(offset + batch_size), :]

                batch_labels = train_labels[offset:(offset + batch_size), :]

                # Prepare the feed dict
                feed_dict = {tf_train_dataset: batch_data,
                             tf_train_labels: batch_labels}

                # run one step of computation
                _, l, predictions = session.run([optimizer, loss, train_prediction],
                                                feed_dict=feed_dict)

                if (step % 500 == 0):
                    print("Minibatch loss at step {0}: {1}".format(step, l))
                    print("Minibatch accuracy: {:.1f}%".format(
                        accuracy(predictions, batch_labels)))
            print("\nTest accuracy: {:.1f}%".format(
                accuracy(test_prediction.eval(), test_labels)))

def CNN(Images_train, Labels_train, Image_test, Label_test):
    batch_size = 128
    num_classes = 10
    epochs = 2

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    x_train = Images_train
    x_test = Image_test

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(Labels_train, num_classes)
    y_test = keras.utils.to_categorical(Label_test, num_classes)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    y_train_pred_ = np.argmax(y_train_pred, axis=1)
    y_test_pred_ = np.argmax(y_test_pred, axis=1)
    # conf_mx_train = confusion_matrix(y_train, y_train_pred_)
    # conf_mx_test = confusion_matrix(y_test, y_test_pred_)
    # print('Confusion matrix (training): \n{0}'.format(conf_mx_train))
    # print('Confusion matrix (test): \n{0}'.format(conf_mx_test))

if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, Train_labels_CNN, Test_labels_CNN = load_data()
    # Softmax(train_images, train_labels, test_images, test_labels)
    # MLP(train_images, train_labels, test_images, test_labels)
    CNN(train_images, Train_labels_CNN, test_images, Test_labels_CNN)