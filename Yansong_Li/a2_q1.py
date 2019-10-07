import numpy as np
import tensorflow as tf
import torch as th
import math
from scipy.misc import derivative
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1.0/(1.0+np.exp(z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def Loss_fuc(x, A, B):
    y = A * x
    u = sigmoid(y)
    v = B * x
    z = A * np.multiply(u,v)
    w = A * z

    norm = []
    for i in w:
        u = np.linalg.norm(i)
        norm.append(u)
    L = np.power(norm, 2)
    loss_val = sum(L)
    return loss_val

def Get_Grad(x, A, B):
    y = A * x
    u = sigmoid(y)
    v = B * x
    z = A * np.multiply(u,v)
    w = A * z
    n = tf.transpose(x, perm=[0, 2, 1])
    sess = tf.Session()
    Trans_X = sess.run(n)
    Grad_A = Trans_X
    Grad_B = Trans_X
    Grad_Y = sigmoid_prime(y)
    Grad_U = np.multiply(A, v)
    Grad_V = np.multiply(A, u)
    Grad_Z = Get_Transpose(A)
    Grad_W = 2 * w
    Grad_L_A = Grad_W * np.multiply(Grad_Z, Grad_U * np.multiply(Grad_A, Grad_Y))
    Grad_L_B = Grad_W * np.multiply(Grad_Z, np.multiply(Grad_V, Grad_B))
    return Grad_L_A, Grad_L_B

def Get_Transpose(x1):
    mv = []
    x1 = np.array(x1)
    for i in x1:
        mv.append(i.T)
    return mv

def get_sum(L):
    sum = np.sum(np.power(L, 2))
    return sum


def backpropgation(grad_A, grad_B, A, B, x):
    alpha = 0.5
    A_update = A + alpha * grad_A
    B_update = B + alpha * grad_B
    loss_updae = Loss_fuc(x, A_update, B_update)
    return loss_updae, A_update, B_update

def sum(L):
    temp = 0
    for i in L:
        temp = temp + i
    return temp


def get_x(N, dim_x):
    np.random.seed(0)
    x = np.random.randint(10, size=(N, dim_x, 1))
    x= np.array(x)
    return x

def find_min(Number, dim):
    loss_list = []
    loop = []
    x = get_x(Number, dim)
    np.random.seed(1)
    A = np.random.randint(0.5, 10, size=[dim, dim])
    B = np.random.randint(0.5, 10, size=[dim, dim])
    it = 0
    A_Sup = []
    B_Sup = []
    while it < Number:
        A_Sup.append(A)
        B_Sup.append(B)
        it = it + 1
    iteration = 0
    while iteration < 1000:
        loss = Loss_fuc(x, A, B)
        Grad_A, Grad_B = Get_Grad(x, A, B)
        loss_new, A_update, B_update= backpropgation(Grad_A, Grad_B, A, B, x)
        A = A_update
        B = B_update
        loss = loss_new
        loss_list.append(loss)
        loop.append(iteration)
        iteration += 1

    return loss, loss_list, loop


loss, loss_list, loop= find_min(10, 5)
plt.figure(figsize=(20, 8))
plt.plot(loop, loss_list)
plt.show()
