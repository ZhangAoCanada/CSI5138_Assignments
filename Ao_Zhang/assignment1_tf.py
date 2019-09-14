import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class PolynomialModel:
    def __init__(self, order, batch_size, learning_rate, regularization):
        self.order = order + 1
        self.X = tf.placeholder(tf.float32, [None, ], "input")
        self.Y = tf.placeholder(tf.float32, [None, ], "label")
        # with tf.variable_scope("weights", reuse=tf.AUTO_REUSE):
        self.Theta = tf.get_variable("parameters" + str(self.order), [self.order, ],
                                    dtype = tf.float32, initializer = tf.glorot_uniform_initializer())
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_lambda = 5e-4

    def debug(self):
        return self.Theta
    
    def Polynomial(self):
        function = tf.constant([0.])
        for i in range(self.order):
            function += self.Theta[i] * (self.X ** i)
        return function

    def getMSE(self):
        prediction = self.Polynomial()
        mean_square_err = tf.reduce_sum((prediction - self.Y) ** 2) / self.batch_size
        return mean_square_err

    def GradientDescent(self):
        prediction = self.Polynomial()
        for i in range(self.order):
            if not self.regularization:
                operator = self.Theta[i].assign(tf.add(self.Theta[i] , self.learning_rate\
                                            * tf.reduce_sum(2 * (self.Y - prediction)\
                                            * (self.X ** i)) / self.batch_size))
            else:
                operator = self.Theta[i].assign(tf.add(tf.add(self.Theta[i] , self.learning_rate\
                                            * tf.reduce_sum(2 * (self.Y - prediction)\
                                            * (self.X ** i)) / self.batch_size)), -2 * self.reg_lambda\
                                            * (self.Theta[i] ** 2))
            yield operator

    def getParameterValues(self):
        return self.Theta


def getData(num_data, variance):
    x = np.random.uniform(0., 1., size = (num_data,))
    y = np.cos(2 * np.pi * x) + np.random.normal(0, variance, size = (num_data,))
    return x, y

def fitData(sess, model, num_train, variance):
    num_test = 2000
    train_x, train_y = getData(num_train, variance)
    test_x, test_y = getData(num_test, variance)

    for op in model.GradientDescent():
        test = sess.run(op, feed_dict = {model.X : train_x, model.Y : train_y})
    
    E_in, para = sess.run([model.getMSE(), model.Theta], feed_dict = {model.X : train_x, model.Y : train_y})
    E_out = sess.run(model.getMSE(), feed_dict = {model.X : test_x, model.Y : test_y})
    return E_in, E_out, para

def experiment(order, num_train, variance, learning_rate, regularization = False):
    M = 50
    num_bias = 2000
    bias_x, bias_y = getData(num_bias, variance)
    E_in_all = []
    E_out_all = []
    theta_all = []
    model = PolynomialModel(order, num_train, learning_rate, regularization)
    init = tf.initializers.global_variables()
    sess = tf.Session()
    for _ in range(M):
        sess.run(init)
        E_in, E_out, theta = fitData(sess, model, num_train, variance)
        E_in_all.append(E_in)
        E_out_all.append(E_out)
        theta_all.append(theta)
    
    E_in_all = np.array(E_in_all)
    E_out_all = np.array(E_out_all)
    theta_all = np.array(theta_all)

    E_in_bar = np.mean(E_in_all, axis = 0)
    E_out_bar = np.mean(E_out_all, axis = 0)
    theta_bar = np.mean(theta_all, axis = 0)

    op = model.Theta.assign(theta_bar)
    sess.run(op)

    E_bias = sess.run(model.getMSE(), feed_dict = {model.X : bias_x, model.Y : bias_y})

    return E_in_bar, E_out_bar, E_bias


N_all = np.array([2, 5, 10, 20, 50, 100, 200])
d_all = np.arange(21)
sigma_all = np.array([0.01, 0.1, 1])

E_in_plot = []
E_out_plot = []
E_bias_plot = []
count = 0

for d in d_all:
    print("Current sequence:\t %d / %d "%(count, len(d_all)))
    N = 10
    sigma = 0.1
    learning_rate = 0.3

    regularization = False

    E_in_bar, E_out_bar, E_bias = experiment(d, N, sigma, learning_rate, regularization)

    E_in_plot.append(E_in_bar)
    E_out_plot.append(E_out_bar)
    E_bias_plot.append(E_bias)

    count += 1

fig = plt.figure(figsize = (8, 8))
ax1 = fig.add_subplot(111)
ax1.plot(d_all, E_in_plot, "r")
ax1.plot(d_all, E_out_plot, "b")
ax1.plot(d_all, E_bias_plot, "g")
plt.show()




