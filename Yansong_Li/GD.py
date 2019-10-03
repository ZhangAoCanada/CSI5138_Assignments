import numpy as np
import matplotlib.pyplot as plt




def getData(sampleNo, sigma):
    mu = 0
    x = np.random.rand(sampleNo)
    np.random.seed(0)
    s = np.random.normal(mu, sigma, sampleNo)
    # plt.hist(s, 30, density=True)
    # plt.show()
    y = np.cos(2*x*np.pi)+ s
    N = list(zip(x,y))
    return N

def getMSE(w, x, y):
    x = np.array(x)
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0/len(x))*np.sum(np.power(error, 2))
    return error, mse

def get_gradient(w, x, y):
    error = getMSE(w, x, y)[0]
    mse = getMSE(w, x, y)[1]
    # w = torch.full([len(w)], len(w))
    # gradient = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones(y.size()))
    # print(gradient)
    gradient = - (1.0/len(x)) * error.dot(x)
    # print(gradient)
    return gradient

# using SGD b407
def fitData(x, y, d, sigma):
    gradient_list = []
    model_order = d
    w = np.random.randn(model_order)
    alpha = 0.1
    epochs = 2000
    decay = 0.95
    batch_size = 10
    iterations = 0
    x = np.array(x)
    x = x.reshape(len(x), 1)
    x = np.power(x, range(model_order))
    x /= np.max(x, axis=0)
    order = np.random.permutation(len(x))
    x = np.array(x)
    y = np.array(y)
    x = x[order]
    y = y[order]
    # while True:
    #     order = np.random.permutation(len(x))
    #     x = np.array(x)
    #     y = np.array(y)
    #     x = x[order]
    #     y = y[order]
    #     b = 0
    #     while b < len(x):
    #         tx = x[b:b+batch_size]
    #         ty = y[b:b+batch_size]
    #         gradient = get_gradient(w, tx, ty)
    #         gradient_list.append(gradient)
    #         error = getMSE(w, x, y)[1]
    #         MSE = getMSE(w, x, y)[1]
    #         w -= alpha * gradient
    #
    #         iterations += 1
    #         b += batch_size
    #     if epochs%100 == 0:
    #         new_error = getMSE(w, x, y)[1]
    #         error_gen =  new_error - error
    #         if abs(new_error - error) < tolerance:
    #
    #             break
    gradient = get_gradient(w, x, y)
    gradient_list.append(gradient)
    MSE = getMSE(w, x, y)[1]
    w = w - alpha * gradient
    MSE_new = getMSE(w, x, y)[1]
    while iterations < epochs:
        w = w - alpha * gradient
        gradient = get_gradient(w, x, y)
        gradient_list.append(gradient)
        if epochs % 100 ==0:
            MSE = MSE_new
            MSE_new  = getMSE(w, x, y)[1]
        iterations += 1
        # alpha = alpha*(decay ** int(epochs/1000))
    Ein = MSE
    Test_Data = getData(2000,sigma)
    X = [i[0] for i in Test_Data]
    Y = [i[1] for i in Test_Data]
    X = np.array(X)
    X = X.reshape(len(X), 1)
    X = np.power(X, range(model_order))
    X /= np.max(X, axis=0)
    Y = np.array(Y)
    Eout = getMSE(w, X, Y)[1]
    return w, Ein, Eout, gradient_list, iterations

def experiment(N,d,sigma):
    gradient_list2 = []
    iterations = 0
    EIN = []
    EOUT = []
    W = []
    while iterations<50:
        M = getData(N, sigma)
        x = [i[0] for i in M]
        y = [i[1] for i in M]
        w, Ein, Eout, gradient_list,iterations_num  = fitData(x, y, d, sigma)
        EIN.append(Ein)
        EOUT.append(Eout)
        W.append(w)
        iterations +=1

    EIN = np.array(EIN)
    EOUT = np.array(EOUT)
    W = np.array(W)
    Ein_mean = np.mean(EIN, axis = 0)
    Eout_mean = np.mean(EOUT, axis = 0)
    M_mean = np.mean(W, axis = 0)
    g = np.array(x)
    h = np.array(y)
    Final_test = getData(2000, sigma)
    x_test = [i[0] for i in Final_test]
    y_test = [i[1] for i in Final_test]
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test), 1)
    x_test = np.power(x_test, range(d))
    x_test /= np.max(x_test, axis=0)
    y_test = np.array(y_test)
    E_bias = getMSE(M_mean, x_test, y_test)[1]
    # f = np.linspace(0, 1, 100)
    # fig = plt.figure(figsize=(8, 8))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(g, h, s = 0.2)
    # for i in range(len(M_mean)):
    #     if i == 0:
    #         y_val = M_mean[0]
    #     else:
    #         y_val += M_mean[i] * (f ** i)
    # ax1.plot(f, y_val, 'r')
    # plt.scatter(g, h)
    # plt.show()
    # length = len(gradient_list)
    # gradient_list = [abs(number) for number in gradient_list]
    # index = np.arange(length)
    # plt.figure(figsize=(20, 8))
    # plt.plot(index, gradient_list)
    # plt.show()
    return Ein_mean, Eout_mean, E_bias


N_all = np.array([2, 5, 10, 20, 50, 100, 200])
d_all = np.arange(21)
sigma_all = np.array([0.01, 0.1, 1])

N = 100
sigma = 0.1
d = 3

Ein_list = []
Eout_list = []
Ebias_list = []
for i in d_all:
    Ein, Eout, Ebias = experiment(N,i,0.1)
    Ein_list.append(Ein)
    Eout_list.append(Eout)
    Ebias_list.append(Ebias)
plt.figure(figsize=(10,10))
plt.plot(d_all, Ein_list, color='green', label='Ein')
plt.plot(d_all, Eout_list, color='red', label='Eout')
plt.plot(d_all, Ebias_list, color = 'yellow', label='Ebias')
plt.xlim(0, 5.5)
plt.ylim(0, 21)
my_x_ticks = np.arange(0, 21, 1)
my_y_ticks = np.arange(0, 5.5, 0.01)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

plt.show()



print(experiment(N,d,0.1)[0])
# print(experiment(N,d,0.1)[1])
# print(experiment(N,d,0.1)[2])

