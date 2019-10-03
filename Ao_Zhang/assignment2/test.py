import numpy as np

a = np.array([[1,2,3], [4,5,6]])
b = np.array([[13,14], [15, 16], [17,18]])

c = np.matmul(a, b)

print(c)