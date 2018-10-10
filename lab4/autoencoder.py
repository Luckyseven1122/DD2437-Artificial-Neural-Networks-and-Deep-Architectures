import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import collections

from plot import Plot
from io_tools import get_training_data, get_testing_data

X, _ = get_training_data()
X_test, X_idx = get_testing_data()
unique_digits = {}

for index, digit in zip(np.arange(X_test.shape[0]),X_idx.reshape(X_idx.shape[0])):
    if(digit not in unique_digits):
        unique_digits[digit] = [index]
    else:
        unique_digits[digit].append(index)

unique_digits = collections.OrderedDict(sorted(unique_digits.items()))
ten_digits_idx = [v[0] for k,v in unique_digits.items()]

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

            # test1_idx, test2_idx = np.random.randint(settings['n_points'], size=2)
            feed_test = {inputs: X_test[ten_digits_idx]}
        i = 0
        for epoch in range(settings['epochs']):
            for batch_idx, batch in enumerate(batches):
                feed = { inputs: batch }

                if(batch_idx % 50 == 0):
                    _, cost = sess.run([optimizer, loss], feed_dict=feed)

                    if(settings['interactive_plot']):
                        r = sess.run(outputs, feed_dict=feed_test)
                        rows, cols = settings['plot_dim']
                        r = np.vstack([X_test[ten_digits_idx],r])
                        plot.custom(data = r,
                                    rows=rows,
                                    cols=cols, 
                                    cost=str(round(cost,5)), 
                                    epoch=str(epoch), 
                                    i=i,
                                    eta=settings['eta'],
                                    hidden_size=settings['hidden_size'])
                        i += 1
                optimizer.run(feed_dict=feed)

            print('Epoch: ' + str(epoch) + ' Cost: ', cost)
            loss_buff.append(cost)

        if(settings['plot_cost']):
            plot.loss(loss_buff)
    return 

''' debuging keep same random numbers'''
np.random.seed(0)

settings = {
    'n_points': X.shape[0],
    'hidden_size': 500,
    'num_batches': 50,
    'epochs': 50,
    'eta': 1e-3,
    'reg_scale': 0.2,
    'interactive_plot': False,
    'plot_dim': (10,2),
    'plot_cost': True
}



settings['batch_size'] = int(X.shape[0] / settings['num_batches'])
print('SETTINGS: ', json.dumps(settings, indent=2))

batches = np.array(np.array_split(X, settings['num_batches']))

train(settings, batches, Y)
