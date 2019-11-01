import numpy as np
import tensorflow as tf

# mnist 784
# cifa 3072
class VAE:
    def __init__(self, input_size, num_hidden_layers, latent_size, dropout = False, BN = False):
        assert isinstance(input_size, int)
        assert isinstance(num_hidden_layers, int)
        assert isinstance(latent_size, int)
        self.input_size = input_size
        self.num_hidden_layers = num_hidden_layers
        self.latent_size = latent_size
        self.X = tf.placeholder(tf.float32, shape = [None, self.input_size])
        self.Z = tf.placeholder(tf.float32, shape = [None, self.latent_size])

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_start = 0.001
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_start, self.global_step, \
                                                        100, 0.96, staircase=True)
        self.dropout = dropout
        self.BN = BN
        self.dropout_rate = 0.2

        # self.weights = {
        #                 'w1' : tf.Variable(tf.glorot_uniform_initializer()((self.input_size, self.n_layer_1))),
        #                 'w2' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_1, self.n_layer_2))),
        #                 'w3' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_2, self.output_size)))
        #                 }
        # self.biases = {
        #                 'b1' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_1,))),
        #                 'b2' : tf.Variable(tf.glorot_uniform_initializer()((self.n_layer_2,))),
        #                 'b3' : tf.Variable(tf.glorot_uniform_initializer()((self.output_size,)))
        #                 }

    def VariableInitializer(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random.normal(shape=size, stddev=xavier_stddev)

    def HiddenLayer(self, input_tensor, in_size, out_size):
        weights = tf.Variable(tf.glorot_uniform_initializer()((in_size, out_size)))
        bias = tf.Variable(tf.glorot_uniform_initializer()((out_size,)))
        output_tensor = tf.matmul(input_tensor, weights) + bias
        if self.BN:
            output_tensor = tf.layers.BatchNormalization()(output_tensor)
        output_tensor = tf.nn.relu(output_tensor)
        if self.dropout:
            output_tensor = tf.layers.dropout(output_tensor, self.dropout_rate)
        return output_tensor

    def LastLayer(self, input_tensor, in_size, out_size):
        weights = tf.Variable(tf.glorot_uniform_initializer()((in_size, out_size)))
        bias = tf.Variable(tf.glorot_uniform_initializer()((out_size,)))
        output_tensor = tf.matmul(input_tensor, weights) + bias
        return output_tensor

    def Encoder(self):
        num_firsthidden = 128
        layer = self.HiddenLayer(self.X, self.input_size, num_firsthidden)
        # for layer_id in range(self.num_hidden_layers):
        #     layer = self.HiddenLayer(layer, num_firsthidden, num_firsthidden//2)
        #     num_firsthidden = num_firsthidden // 2
        #     if num_firsthidden <= 128:
        #         break
        encode_mean = self.LastLayer(layer, num_firsthidden, self.latent_size)
        encode_variance = tf.nn.softplus(self.LastLayer(layer, num_firsthidden, self.latent_size))
        return encode_mean, encode_variance

    def Decoder(self, x):
        num_firsthidden = 128
        layer = self.HiddenLayer(x, self.latent_size, num_firsthidden)
        # for layer_id in range(self.num_hidden_layers):
        #     layer = self.HiddenLayer(layer, num_firsthidden, num_firsthidden*2)
        #     num_firsthidden = num_firsthidden * 2
        #     if num_firsthidden >= 512:
        #         break
        layer = self.LastLayer(layer, num_firsthidden, self.input_size)
        logits = layer
        prob = tf.nn.sigmoid(logits)
        return logits, prob

    def SampleLatent(self, mean, variance):
        eps = tf.random.normal(shape = tf.shape(mean))
        return mean + variance * eps

    def Loss(self,):
        mean, variance = self.Encoder()
        sampling = self.SampleLatent(mean, variance)
        sample_logits, sample_prob = self.Decoder(sampling)

        # marginal_likelihood = tf.reduce_sum(x * tf.log(y) + (1 - x) * tf.log(1 - y), 1)

        # E(log p|z (x_i | z_i))
        reconstruction_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                                            logits=sample_logits, labels=self.X), axis=1)
        # KL(q z|x (.|x_i) || p Z), since the variance we got is actually log(variance)
        # therefore, we move everthing inside log
        # KL_loss = 0.5 * tf.reduce_sum(tf.exp(variance) + mean**2 - 1. - variance, axis=1)
        KL_loss = 0.5 * tf.reduce_sum(tf.square(mean) + tf.square(variance) - \
                        tf.log(1e-8 + tf.square(variance)) - 1, 1)
        # vae loss function
        vae_loss = tf.reduce_mean(reconstruction_loss - KL_loss)
        return vae_loss

    def TrainModel(self):
        loss = self.Loss()
        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        learning_operation = optimizer.minimize(loss, global_step = self.global_step)
        return learning_operation


# loss_graph_name = "loss"
#     acc_graph_name = "accuracy"
#     summary_loss = tf.summary.scalar(loss_graph_name, loss)
#     streaming_accuracy, streaming_accuracy_update = tf.contrib.metrics.streaming_mean(accuracy)
#     summary_accuracy = tf.summary.scalar(acc_graph_name, streaming_accuracy)
#     # summary_accuracy_straight = tf.summary.scalar(acc_graph_name, accuracy)

#     train = model.TrainModel()

#     # initialization
#     init = tf.global_variables_initializer()

#     # GPU settings
#     gpu_options = tf.GPUOptions(allow_growth=True)

#     with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
#         summaries_train = 'logs/train/'
#         summaries_test = 'logs/test/'
#         folder_name = FolderName(mode, dropout, BN)
#         train_writer = tf.summary.FileWriter(summaries_train + folder_name, sess.graph)
#         test_writer = tf.summary.FileWriter(summaries_test + folder_name, sess.graph)
#         summary_acc = tf.Summary()
#         sess.run(init)
#         for each_epoch in tqdm(range(epoches)):
#             for each_batch_train in range(hm_batches_train):
#                 X_train_batch = X_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]
#                 Y_train_batch = Y_train[each_batch_train*batch_size: (each_batch_train+1)*batch_size]

#                 _, loss_val, summary_l, steps = sess.run([train, loss, summary_loss, model.global_step], \
#                                                                     feed_dict = {model.X : X_train_batch, \
#                                                                                 model.Y : Y_train_batch})

#                 train_writer.add_summary(summary_l, steps)
                
#                 """
#                 When GPU memory is not enough
#                 """
#                 sess.run(tf.local_variables_initializer())
#                 for each_batch_test in range(hm_batches_test):
#                     X_test_batch = X_test[each_batch_test*batch_size: (each_batch_test+1)*batch_size]
#                     Y_test_batch = Y_test[each_batch_test*batch_size: (each_batch_test+1)*batch_size]
#                     sess.run([streaming_accuracy_update], feed_dict = {model.X : X_test_batch, \
#                                                             model.Y : Y_test_batch})

#                 summary_a = sess.run(summary_accuracy)
#                 test_writer.add_summary(summary_a, steps)




