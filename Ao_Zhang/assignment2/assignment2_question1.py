"""
CSI 5138: Assignment 2 ----- Question 1
Student:            Ao   Zhang
Student Number:     0300039680
"""
##### for plotting through X11 #####
import matplotlib
matplotlib.use("tkagg")
##### set specific gpu #####
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
##### other dependencies #####
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from multiprocessing import Process

class QuestionOne:
    def __init__(self, K):
        """
        Args:
            k               ->              dimension of input vector X
        """
        self.K = K
        self.X = tf.placeholder(tf.float32, shape = (None, self.K, 1))
        self.A = tf.Variable(tf.glorot_uniform_initializer()((1, self.K, self.K)))
        self.B = tf.Variable(tf.glorot_uniform_initializer()((1, self.K, self.K)))

    ###################################################################
    # define all functions and its relative gradients
    ###################################################################
    def FuncLinear(self, coef, var):
        """
        Function: y = A * x
        """
        return tf.matmul(coef, var)

    def GradientLinear(self, coef):
        """
        Function: grad(y) = A
        """
        return coef

    def Sigmoid(self, input_var):
        """
        Function: Sigmoid
        """
        return 1. / (1. + tf.exp( - input_var))

    def GradientSigmoid(self, input_var):
        """
        Function: grad(sigmoid) = sigmoid * (1 - sigmoid)
        """
        return self.Sigmoid(input_var) * (1 - self.Sigmoid(input_var))

    def FuncMultiplication(self, coef, input_var1, input_var2):
        """
        Function: y = A * (u * v)
        """
        return tf.matmul(coef, (input_var1 * input_var2))

    def GradientMultiplication(self, coef, input_var1, input_var2):
        """
        Function: grad(y) = Au + Av
        """
        return tf.add(tf.matmul(coef, input_var1), tf.matmul(coef, input_var2))

    def EuclideanNorm(self, input_var):
        """
        Function: Euclidean Norm(X)
        """
        return tf.reduce_sum(tf.square(input_var), axis = 1, keepdims=True)
    
    def GradientEuclideanNorm(self, input_var):
        """
        Function: 2*x_1 + 2*x_2 + ... + 2*x_n
        """
        return 2 * tf.reduce_sum(input_var, axis = 1, keepdims=True)

    ###################################################################
    # calculate the forward graph, gradient graph and dual gradient
    ###################################################################
    def ForwardGraph(self):
        """
        Function: Calculate loss function from input X
        """
        y = self.FuncLinear(self.A, self.X)
        u = self.Sigmoid(y)
        v = self.FuncLinear(self.B, self.X)
        z = self.FuncMultiplication(self.A, u, v)
        omega = self.FuncLinear(self.A, z)
        loss = self.EuclideanNorm(omega)
        return loss
    
    def GradientGraph(self):
        """
        Function: Calculate forward gradient graph
        """
        grad_y = self.GradientLinear(self.A)
        grad_u = self.GradientSigmoid(grad_y)
        grad_v = self.GradientLinear(self.B)
        grad_z = self.GradientMultiplication(self.A, grad_u, grad_v)
        grad_omega = self.GradientLinear(self.A) * grad_z
        grad_loss = self.GradientEuclideanNorm(grad_omega)
        return grad_loss

    def DualGradient(self):
        """
        Function: Dual gradient = gradient.transpose()
        """
        gradient = self.GradientGraph()
        # Since the first dimension is the batch size, we should keep it in the first column
        dual_grad = tf.transpose(gradient, perm = [0, 2, 1])
        return dual_grad

    def BackPropGradientDescent(self):
        """
        Function: Apply GD based on back propagation
        """
        


if __name__ == "__main__":
    K = 5

    Q_one =  QuestionOne(K)
    X_data = np.random.randint(10, size = (1, K, 1))


    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    init = tf.initializers.global_variables()

    sess.run(init)

    loss = Q_one.ForwardGraph()
    gradient = Q_one.DualGradient()
    test1, test2 = sess.run([loss, gradient], feed_dict = {Q_one.X: X_data})
    print(test1.shape)
    print(test2.shape)


    



