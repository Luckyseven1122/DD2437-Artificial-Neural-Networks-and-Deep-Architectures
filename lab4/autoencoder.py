import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json

from plot import Plot
from io_tools import get_training_data, get_testing_data

X, _ = get_training_data()
Y = X[0:2000]
X = X[0:2000]
plot = Plot()
# -- plot examples --
# plot.one(train[0])
# plot.nine(train[0:9])
# plot.custom(data = train[0:15],rows = 3, cols = 5)

def autoencoder(inputs, hidden_size, reg_scale):
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

    x = tf.contrib.layers.fully_connected(
            inputs,
            hidden_size,
            weights_regularizer=regularizer,
            activation_fn=tf.nn.sigmoid)

    decode = tf.contrib.layers.fully_connected(x, 784, activation_fn=None)
    return decode


def train(settings, batches, Y):
    loss_buff = []

    inputs = tf.placeholder(tf.float32, shape=[None, 784])
    outputs = autoencoder(inputs = inputs,
                          hidden_size = settings['hidden_size'],
                          reg_scale= settings['reg_scale'])

    loss = tf.reduce_mean(tf.square(inputs - outputs))
    optimizer = tf.train.AdamOptimizer(learning_rate=settings['eta']).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        if(settings['interactive_plot']):
            plt.show() 
            plt.figure(figsize= settings['plot_dim'])

        for epoch in range(settings['epochs']):
            for batch_idx, batch in enumerate(batches):
                feed = { inputs: batch }
                if(batch_idx % 30 == 0):
                    _, cost = sess.run([optimizer, loss], feed_dict=feed)

                    if(settings['interactive_plot']):
                        Y = Y[0:settings['batch_size']]
                        r = sess.run(outputs, feed_dict={inputs: Y})
                        rows, cols = settings['plot_dim']
                        plot.custom(data = r,rows=rows,cols=cols)

                optimizer.run(feed_dict=feed)

            print('Epoch: ' + str(epoch) + ' Cost: ', cost)
            loss_buff.append(cost)

        if(settings['plot_cost']):
            plot.loss(loss_buff)
    return 


settings = {
    'hidden_size': 500,
    'num_batches': 50,
    'epochs': 50,
    'eta': 1e-3,
    'reg_scale': 0.9,
    'interactive_plot': True,
    'plot_dim': (10,4),
    'plot_cost': True
}



settings['batch_size'] = int(X.shape[0] / settings['num_batches'])
print('SETTINGS: ', json.dumps(settings, indent=2))

batches = np.array(np.array_split(X, settings['num_batches']))

train(settings, batches, Y)
