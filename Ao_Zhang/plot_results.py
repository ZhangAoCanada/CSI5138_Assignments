import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np


E_in = np.load("saved_results/N_10_d_all_sig_0.1_Ein.npy")
E_out = np.load("saved_results/N_10_d_all_sig_0.1_Eout.npy")
E_bias = np.load("saved_results/N_10_d_all_sig_0.1_Ebias.npy")

x = np.arange(21)


fig = plt.figure(figsize = (8, 8))
ax1 = fig.add_subplot(111)
ax1.plot(x, E_in, "r")
ax1.plot(x, E_out, "g")
ax1.plot(x, E_bias, "b")
plt.show()