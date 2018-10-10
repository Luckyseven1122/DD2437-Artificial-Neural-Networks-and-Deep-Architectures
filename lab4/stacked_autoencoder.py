from autoencoder import network
import tensorflow as tf
from io_tools import get_training_data, get_testing_data

class Stacked_ae:
    def __init__(self):
        self.train_X, self.train_Y = get_training_data()
        self.test_X, self.test_Y = get_testing_data()

        self.stack()

    def stack(self):
        labels = self.train_Y
        idx = []
        counter = 0
        for i in range(labels.size):
            if labels[i] == counter:
                idx.append(i)
                counter += 1

        settings = {
                'hidden_size': 120,
                'num_batches': 50,
                'epochs': 1,
                'eta': 1e-3,
                'reg_scale': 0.9,
                'interactive_plot': False,
                'plot_dim': (12,10),
                'plot_cost': False,
                'plot_weights': False,
                'plot_numbers': False,
                'number_idx': idx,
                'store_weights': True,
                'calculate_avg_sparseness': True
            }

        settings['batch_size'] = int(self.train_X.shape[0] / settings['num_batches'])

        self.weights, self.bias = {},{}

        w1,b1 = network('layer1', settings, hidden_size = 500).run()
        w2,b2 = network('layer2', settings, hidden_size = 200).run()

        self.weights['w1'], self.weights['w2'] = w1, w2
        self.bias['b1'], self.bias['b2']  = b1, b2
        
        inputs = tf.placeholder(tf.float32, shape=[None, 784])

        encoded = self.encode(inputs)
        decoded = self.decode(encoded)

        loss = tf.reduce_mean(tf.square(self.train_X - decoded))
        optimizer = tf.train.AdamOptimizer(learning_rate=settings['eta']).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

    def encode(self, x):
        l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['w1']), self.bias['b1']))
        encode = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['w2']), self.bias['b2']))
        return encode 
    
    def decode(self, x):
        l1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['w2']), self.bias['b2']))
        decode = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['w1']), self.bias['b1']))
        return decode 
        
   

