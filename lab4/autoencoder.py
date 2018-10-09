import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import imp

from plot import Plot

# -- plot examples --
# plot.one(train[0])
# plot.nine(train[0:9])
# plot.custom(data = train[0:15],rows = 3, cols = 5)

class network:
    def __init__(self, X, Y, test_X, test_Y):
        self.plot = Plot()
        self.train_X = X
        self.train_Y = Y
        self.test_X = test_X
        self.test_Y = test_Y

    def autoencoder(self, inputs, hidden_size, reg_scale):
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

        x = tf.contrib.layers.fully_connected(
                inputs,
                hidden_size,
                weights_regularizer=regularizer,
                activation_fn=tf.nn.sigmoid)

        decode = tf.contrib.layers.fully_connected(x, 784, activation_fn=None)
        return decode


    def train(self, settings, batches, Y):
        inputs = tf.placeholder(tf.float32, shape=[None, 784])
        outputs = self.autoencoder(inputs = inputs,
                              hidden_size = settings['hidden_size'],
                              reg_scale= settings['reg_scale'])

        loss = tf.reduce_mean(tf.square(inputs - outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=settings['eta']).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(settings['epochs']):
                for batch_idx, batch in enumerate(batches):
                    feed = { inputs: batch }
                    if(batch_idx % 30 == 0):
                        _, cost = sess.run([optimizer, loss], feed_dict=feed)
                    optimizer.run(feed_dict=feed)

                print('Epoch: ' + str(epoch) + ' Cost: ', cost)
            Y = Y[0:settings['batch_size']]
            reconstructed = sess.run(outputs, feed_dict={inputs: Y})

            self.plot.custom(data = reconstructed, rows = 4, cols = 10)
        return

    def run(self):

        X = self.train_X
        Y = X[0:2000]
        X = X[0:2000]

        settings = {
            'hidden_size': 500,
            'num_batches': 50,
            'epochs': 9,
            'eta': 1e-3,
            'reg_scale': 0.9
        }

        settings['batch_size'] = int(X.shape[0] / settings['num_batches'])
        print('SETTINGS: ', json.dumps(settings, indent=2))

        batches = np.array(np.array_split(X, settings['num_batches']))

        self.train(settings, batches, Y)
