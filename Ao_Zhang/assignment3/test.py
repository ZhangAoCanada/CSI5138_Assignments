import tensorflow as tf

params = tf.constant([10,20,30,40])
ids = tf.constant([0,1,2,3])
with tf.Session() as sess:
    print(tf.nn.embedding_lookup(params,ids).eval())