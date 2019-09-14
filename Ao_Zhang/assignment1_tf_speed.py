import matplotlib
matplotlib.use("tkagg")
import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

class PolynomialModel:
    def __init__(self, order, batch_size, learning_rate, regularization):
        self.order = order + 1
        self.X = tf.placeholder(tf.float32, [None, 1], "input")
        self.Y = tf.placeholder(tf.float32, [None, 1], "label")
        self.X_poly = self.RebuildInput()
        self.Theta = tf.get_variable("parameters" + str(self.order), [self.order, 1],
                                    dtype = tf.float32, initializer = tf.glorot_uniform_initializer())
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.reg_lambda = 5e-4

    def RebuildInput(self):
        for i in range(self.order):
            if i == 0:
                X_poly_form = self.X ** i
            else:
                X_poly_form = tf.concat([X_poly_form, (self.X ** i)], axis = -1)
        return X_poly_form
    
    def Polynomial(self):
        return tf.matmul(self.X_poly, self.Theta)

    def getMSE(self):
        prediction = self.Polynomial()
        mean_square_err = tf.reduce_sum((prediction - self.Y) ** 2) / self.batch_size
        return mean_square_err

    def GradientDescent(self):
        prediction = self.Polynomial()
        learning = tf.add(self.Theta , self.learning_rate * 2 * \
                        tf.matmul(tf.transpose(self.X_poly), (self.Y - prediction)) / \
                        self.batch_size)
        if self.regularization:
            learning = tf.add(learning, -2 * self.learning_rate * self.reg_lambda * self.Theta)
        operator = self.Theta.assign(learning)
        return operator


def getData(num_data, variance):
    x = np.random.uniform(0., 1., size = (num_data,))
    y = np.cos(2 * np.pi * x) + np.random.normal(0, variance, size = (num_data,))
    x = np.expand_dims(x, axis = -1)
    y = np.expand_dims(y, axis = -1)
    return x, y

def fitData(sess, model, num_train, variance):
    num_test = 2000
    train_x, train_y = getData(num_train, variance)
    test_x, test_y = getData(num_test, variance)

    op = model.GradientDescent()
    test = sess.run(op, feed_dict = {model.X : train_x, model.Y : train_y})
    
    E_in, para = sess.run([model.getMSE(), model.Theta], feed_dict = {model.X : train_x, model.Y : train_y})
    E_out = sess.run(model.getMSE(), feed_dict = {model.X : test_x, model.Y : test_y})
    return E_in, E_out, para

def experiment(sess, order, num_train, variance, learning_rate, regularization = False):
    M = 50
    num_bias = 3000
    bias_x, bias_y = getData(num_bias, variance)
    E_in_all = []
    E_out_all = []
    theta_all = []
    model = PolynomialModel(order, num_train, learning_rate, regularization)
    init = tf.initializers.global_variables()
    # gpu_options = tf.GPUOptions(allow_growth=True)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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

def main(current_test):
    N_all = np.array([2, 5, 10, 20, 50, 100, 200])
    d_all = np.arange(21)
    sigma_all = np.array([0.01, 0.1, 1])

    all_changing_data = [N_all, d_all, sigma_all]

    switcher = {"test_N": 0,
                "test_d": 1,
                "test_sigma": 2}

    test_num = switcher[current_test]

    E_in_plot = []
    E_out_plot = []
    E_bias_plot = []

    total_len = len(all_changing_data[test_num])

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    for ind in tqdm(range(total_len)):
        if test_num == 0:
            N = N_all[ind]
            d = 10
            sigma = 0.1
        elif test_num == 1:
            N = 100
            d = d_all[ind]
            sigma = 0.1
        else:
            N = 50
            d = 10
            sigma = sigma_all[ind]

        learning_rate = 1
        regularization = False

        E_in_bar, E_out_bar, E_bias = experiment(sess, d, N, sigma, learning_rate, regularization)

        E_in_plot.append(E_in_bar)
        E_out_plot.append(E_out_bar)
        E_bias_plot.append(E_bias)

    E_in_plot = np.array(E_in_plot)
    E_out_plot = np.array(E_out_plot)
    E_bias_plot = np.array(E_bias_plot)

    if test_num == 0:
        N = "all"
    elif test_num == 1:
        d = "all"
    else:
        sigma = "all"

    np.save("saved_results/" + current_test + "N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma_all) + "_Ein.npy", E_in_plot)
    np.save("saved_results/" + current_test + "N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma_all) + "_Eout.npy", E_out_plot)
    np.save("saved_results/" + current_test + "N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma_all) + "_Ebias.npy", E_bias_plot)
    # fig = plt.figure(figsize = (8, 8))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(d_all, E_in_plot, "r")
    # ax1.plot(d_all, E_out_plot, "b")
    # ax1.plot(d_all, E_bias_plot, "g")
    # plt.show()

if __name__ == "__main__":

    current_test = "test_d"
    p = Process(target = main, args=(current_test,))
    p.start()
    p.join()



