import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
        #tf = imp.reload(tf)

    def __del__(self):
        tf.reset_default_graph()

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
        loss_buff = []

        inputs = tf.placeholder(tf.float32, shape=[None, 784])
        outputs = self.autoencoder(inputs = inputs,
                              hidden_size = settings['hidden_size'],
                              reg_scale= settings['reg_scale'])

        loss = tf.reduce_mean(tf.square(inputs - outputs))
        optimizer = tf.train.AdamOptimizer(learning_rate=settings['eta']).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if(settings['interactive_plot']):
                #plt.show()
                plt.figure(figsize=settings['plot_dim'])

            for epoch in range(settings['epochs']):
                for batch_idx, batch in enumerate(batches):
                    feed = { inputs: batch }
                    if(batch_idx % 30 == 0):
                        _, cost = sess.run([optimizer, loss], feed_dict=feed)

                        if(settings['interactive_plot']):
                            Y = Y[0:settings['batch_size']]
                            r = sess.run(outputs, feed_dict={inputs: Y})
                            rows, cols = settings['plot_dim']
                            self.plot.custom(data = r,rows=rows,cols=cols)

                        if(settings['plot_weights']):
                            W = [w for w in tf.trainable_variables() if w.name == 'fully_connected_1/weights:0'][0]
                            W = sess.run(W.value())
                            hidden = W.shape[0]
                            nodes = W.shape[1]
                            rows, cols = settings['plot_dim']
                            assert rows*cols == hidden
                            self.plot.custom(W, rows, cols, save={'path': epoch})

                    optimizer.run(feed_dict=feed)

                print('Epoch: ' + str(epoch) + ' Cost: ', cost)
                loss_buff.append(cost)

            if(settings['plot_cost']):
                self.plot.loss(loss_buff)

            if(settings['plot_weights']):
                W = [w for w in tf.trainable_variables() if w.name == 'fully_connected_1/weights:0'][0]
                W = sess.run(W.value())
                hidden = W.shape[0]
                nodes = W.shape[1]
                rows, cols = settings['plot_dim']
                assert rows*cols == hidden
                self.plot.custom(W, rows, cols)


    def run(self):
        X = self.train_X
        Y = X[0:2000]
        X = X[0:2000]

        settings = {
            'hidden_size': 50,
            'num_batches': 50,
            'epochs': 100,
            'eta': 1e-3,
            'reg_scale': 0.9,
            'interactive_plot': False,
            'plot_dim': (5,10),
            'plot_cost': False,
            'plot_weights': True,
        }

        settings['batch_size'] = int(X.shape[0] / settings['num_batches'])
        print('SETTINGS: ', json.dumps(settings, indent=2))

        batches = np.array(np.array_split(X, settings['num_batches']))

        self.train(settings, batches, Y)
