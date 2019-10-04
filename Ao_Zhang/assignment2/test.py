import tensorflow as tf

a = tf.constant([[1], [2]])

b = a.T

sess = tf.Session()

bbb = sess.run(b)
print(bbb.shape)