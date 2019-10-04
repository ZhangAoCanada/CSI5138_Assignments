import tensorflow as tf


a = tf.constant([[1.], [0.]])

c = tf.ones([3., 1., 1.])

# b = tf.tile(a, [3,1])
b = c * a

sess = tf.Session()
bb = sess.run(b)

print(bb.shape)