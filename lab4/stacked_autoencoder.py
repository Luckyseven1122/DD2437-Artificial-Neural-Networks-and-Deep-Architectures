from autoencoder import network
import tensorflow as tf
import numpy as np
from io_tools import get_training_data, get_testing_data

tf.logging.set_verbosity(tf.logging.INFO)

class Stacked_ae:
    def __init__(self):
        self.train_X, self.train_Y = get_training_data()
        self.test_X, self.test_Y = get_testing_data()

        self.stack()

    def one_hot_labels(self, y):
        labels = np.array([np.zeros(10)]*y.shape[0])
        for i, l in enumerate(y):
            labels[i][l[0]] = 1
        return labels

    def __del__(self):
        tf.reset_default_graph()

    def stack(self):
        labels = self.train_Y

        settings = {
                'hidden_size': 120,
                'num_batches': 65,
                'epochs': 50,
                'eta': 1e-3,
                'reg_scale': 0.5,
                'interactive_plot': False,
                'plot_dim': (12,10),
                'plot_cost': False,
                'plot_weights': False,
                'plot_numbers': False,
                'store_weights': True,
                'calculate_avg_sparseness': True
            }

        settings['batch_size'] = int(self.train_X.shape[0] / settings['num_batches'])

        # 40, 30, 20, 15
        l1_size, l2_size, l3_size= 500, 50, 10
        w1,b1 = network(layer_name='l1', 
                               data=self.train_X, 
                               settings=settings, hidden_size = l1_size, n_input = 784).run()

        w2,b2= network(layer_name='l2', 
                               data=w1, 
                               settings=settings, hidden_size = l2_size, n_input = l1_size).run()

        w3,b3= network(layer_name='l3', 
                               data=w2, 
                               settings=settings, hidden_size = l3_size, n_input = l2_size).run()

        # w4,b4= network(layer_name='l4', 
        #                        data=w3, 
        #                        settings=settings, hidden_size = l4_size, n_input = l3_size).run()

        inputs = tf.placeholder(tf.float32, shape=[None, 784])

        self.w1_placeholder = tf.Variable(tf.placeholder_with_default(input = w1, shape=[784, l1_size]))
        self.w2_placeholder = tf.Variable(tf.placeholder_with_default(input = w2, shape=[l1_size,l2_size]))
        self.w3_placeholder = tf.Variable(tf.placeholder_with_default(input = w3, shape=[l2_size,l3_size]))
        # self.w4_placeholder = tf.Variable(tf.placeholder_with_default(input = w4, shape=[l3_size,l4_size]))

        self.b1_placeholder = tf.Variable(tf.placeholder_with_default(input = b1, shape=[l1_size]))
        self.b2_placeholder = tf.Variable(tf.placeholder_with_default(input = b2, shape=[l2_size]))
        self.b3_placeholder = tf.Variable(tf.placeholder_with_default(input = b3, shape=[l3_size]))
        # self.b4_placeholder = tf.Variable(tf.placeholder_with_default(input = b4, shape=[l4_size]))

        encoded = self.encode(inputs)

        true_labels = tf.placeholder(tf.float32, shape=[None,10])

        one_hot_labels = self.one_hot_labels(self.train_Y)
        one_hot_labels_test = self.one_hot_labels(self.test_Y)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=encoded, labels=true_labels))

        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
        optimizer = tf.train.RMSPropOptimizer(1e-4).minimize(loss)

        batches = np.array(np.array_split(self.train_X, settings['num_batches']))
        batches_labels = np.array(np.array_split(one_hot_labels, settings['num_batches']))
        
        settings['epochs'] = 400
        costs_test, costs_train = [], []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(settings['epochs']):
                for batch_idx, batch in enumerate(batches):
                    feed = { inputs: batch, true_labels: batches_labels[batch_idx]}
                    _, cost = sess.run([optimizer, loss], feed_dict=feed)
                



                probs,c = sess.run([encoded, loss], feed_dict={inputs: self.test_X, true_labels: one_hot_labels_test})
                
                acc = self.accuracy(probs) 
                print('Epoch:\t', np.round(epoch,3) ,' Cost:\t', np.round(c,5), ' Accuracy:\t', np.round(acc,4))


                probs_test,c_t = sess.run([encoded, loss], feed_dict={inputs: self.train_X, true_labels: one_hot_labels})

                
                costs_test.append(c)
                costs_train.append(c_t)

            # probs = sess.run(encoded, feed_dict={inputs: self.test_X})
        np.save('cost_layer_500_50_15_train_BEST', np.array(costs_train))
        np.save('cost_layer_500_50_15_test_BEST', np.array(costs_test))
    
        # ACC second, 250: 83 %, 100: 86 %, 50: 86 %, 30: 86 %
        # ACC third 15 = 86 % 20 = 85 %, 30 = 85 %, 40: 85 %
    def confusion(self, probs):
        predictions = np.argmax(probs, axis=1)
        real = self.test_Y.reshape(self.test_Y.shape[0])
        confusion = np.zeros((10,10))

        for pred, exp in zip(predictions, real):
            confusion[pred][exp] += 1
        print(confusion)


    def accuracy(self, probs):
        pred = np.argmax(probs, axis=1)
        real = self.test_Y.reshape(self.test_Y.shape[0])
        diff = 0
        for p, r in zip(pred, real):
            if p != r:
                diff += 1
        n = self.test_Y.shape[0]
        acc = (n - diff) / n
        return acc

    
    def encode(self, x):
        l1 = self.g(tf.add(tf.matmul(x, self.w1_placeholder), self.b1_placeholder))
        l2 = self.g(tf.add(tf.matmul(l1, self.w2_placeholder), self.b2_placeholder))
        # l3 = self.g(tf.add(tf.matmul(l2, self.w3_placeholder), self.b3_placeholder))
        encoded = tf.add(tf.matmul(l2, self.w3_placeholder), self.b3_placeholder)
        return encoded 
    
    def g(self, x):
       return tf.nn.sigmoid(x)
