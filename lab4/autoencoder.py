import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import imp

from io_tools import get_training_data, get_testing_data
from plot import Plot

# -- plot examples --
# plot.one(train[0])
# plot.nine(train[0:9])
# plot.custom(data = train[0:15],rows = 3, cols = 5)

class network:
    def __init__(self,data, layer_name, settings, hidden_size, n_input):
        self.data = data
        self.settings = settings
        self.settings['hidden_size'] = hidden_size 
        self.n_input = n_input

        self.layer_name = layer_name
    
    def __del__(self):
        tf.reset_default_graph()

    def autoencoder(self, inputs, hidden_size, reg_scale):

        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

        x = tf.contrib.layers.fully_connected(
                inputs,
                num_outputs=hidden_size,
                weights_initializer=tf.initializers.random_normal,
                weights_regularizer=regularizer,
                activation_fn=tf.nn.sigmoid)

        x = tf.nn.dropout(x, 0.6)
        decode = tf.contrib.layers.fully_connected(x, self.n_input, activation_fn=None)
        return decode


    def train(self, settings, batches):
        inputs = tf.placeholder(tf.float32, shape=[None, self.n_input])

        outputs = self.autoencoder(inputs = inputs,
                              hidden_size = settings['hidden_size'],
                              reg_scale= settings['reg_scale'])

        loss = tf.reduce_mean(tf.square(inputs - outputs))

        # optimizer = tf.train.AdamOptimizer(learning_rate=settings['eta']).minimize(loss)
        optimizer = tf.train.RMSPropOptimizer(1e-3).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(settings['epochs']):
                for batch_idx, batch in enumerate(batches):
                    feed = { inputs: batch }
                    optimizer.run(feed_dict=feed)

            layer = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            layer = sess.run(layer)
            weight,bias = layer[0], layer[1]
            print('done: ', self.layer_name)
        return weight, bias


    def single_run(self):
        X = self.data
        batches = np.array(np.array_split(X, self.settings['num_batches']))

        return self.train(self.settings, batches)


    def run(self):
        return self.single_run()
        #self.do_loss()

