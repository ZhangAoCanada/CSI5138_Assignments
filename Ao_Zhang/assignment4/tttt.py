import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import numpy as np


a = np.arange(100)

b = 2 * a

plt.plot(a, b)
plt.show()