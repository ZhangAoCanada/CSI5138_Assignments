# import tensorflow as tf
import os

print(os.path.isdir("test"))
if os.path.isdir("test"):
    print("exits")
else:
    os.mkdir("test")