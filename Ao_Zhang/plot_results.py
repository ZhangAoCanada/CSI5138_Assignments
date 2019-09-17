import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np

current_test = "test_d"

# set a switch case for switch the questions quickly
switcher = {"test_N": 0,
            "test_d": 1,
            "test_sigma": 2}
test_num = switcher[current_test]

if test_num == 0:
    N = "all"
    d = 5
    sigma = 0.1
    leng = 7
elif test_num == 1:
    N = 20
    d = "all"
    sigma = 0.1
    leng = 21
else:
    N = 100
    d = 10
    sigma = "all"
    leng = 3


E_in = np.load("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + "_Ein.npy")
E_out = np.load("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + "_Eout.npy")
E_bias = np.load("results/" + current_test + "_N_"+ str(N) +"_d_" + str(d) + "_sig_" + str(sigma) + "_Ebias.npy")

x = np.arange(leng)


fig = plt.figure(figsize = (8, 8))
ax1 = fig.add_subplot(111)
ax1.plot(x, E_in, "r")
ax1.plot(x, E_out, "g")
ax1.plot(x, E_bias, "b")
# ax1.set_ylim((0, 2))
plt.show()