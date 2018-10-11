import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import json, imp, re, os.path

from plot import Plot

# -- plot examples --
# plot.one(train[0])
# plot.nine(train[0:9])
# plot.custom(data = train[0:15],rows = 3, cols = 5)

class network:
    def one_hot_labels(self, y):
        labels = np.array([np.zeros(10)]*y.shape[0])
        for i, l in enumerate(y):
            labels[i][l[0]] = 1
        return labels

    def __init__(self, X, Y, test_X, test_Y):
        self.plot = Plot()
        self.train_X = X
        self.train_Y = self.one_hot_labels(Y)
        self.test_X = test_X
        self.test_Y = self.one_hot_labels(test_Y)

        #tf = imp.reload(tf)

    def __del__(self):
        tf.reset_default_graph()

    def _config_to_path(self, conf):
        N = 'N=' + str(conf['hidden']) # int 120,
        ETA = '_eta=' + str(conf['eta']) #float 1e-3,
        LAMBDA = '_lambda=' + str(conf['reg']) #float 0.0,
        N_BATCHES = '_n_batches=' + str(conf['num_batches']) #int 50,
        EPOCHS = '_epochs=' + str(conf['epochs']) # int 50
        filename = N + ETA + LAMBDA + N_BATCHES + EPOCHS
        path_name = re.sub(r'[^\w\s\/-=]','', filename)
        return path_name

    def autoencoder(self, inputs, input_size, hidden_size, reg_scale):
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

        x = tf.contrib.layers.fully_connected(
                inputs,
                hidden_size,
                weights_initializer=tf.initializers.random_normal,
                weights_regularizer=regularizer,
                activation_fn=tf.nn.sigmoid)

        decode = tf.contrib.layers.fully_connected(x, input_size, activation_fn=None)
        return decode

    def classifier(self, inputs, output_size, weights, biases, reg_scale):
        regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)

        x = None
        for i in range(0,len(weights)):
            x = tf.contrib.layers.fully_connected(
                    inputs=inputs if i < 1 else x,
                    num_outputs=weights[i].shape[1],
                    weights_initializer=tf.initializers.random_normal,
                    weights_regularizer=regularizer,
                    activation_fn=tf.nn.sigmoid)

        # find all variables
        W = [w for w in tf.trainable_variables() if 'weight' in w.name]
        B = [b for b in tf.trainable_variables() if 'biases' in b.name]

        assert len(W) == len(weights)
        assert len(B) == len(biases)

        # add pretrained weights
        for i in range(len(W)):
            W[i].assign(weights[i], True)
            B[i].assign(biases[i], True)

        output = tf.contrib.layers.fully_connected(x, output_size)
        return output

    def train(self, settings, batches, Y):
        loss_buff = []
        loss_test_buff = []
        sparse = []

        inputs = tf.placeholder(tf.float32, shape=[None, settings['input_size']])
        labels = tf.placeholder(tf.float32, shape=[None, 10])

        if settings['outputs'] == 'autoencoder':
            outputs = self.autoencoder(inputs = inputs,
                                       input_size = settings['input_size'],
                                       hidden_size = settings['hidden'],
                                       reg_scale= settings['reg'])

            loss = tf.reduce_mean(tf.square(inputs - outputs))
        else:
            outputs = self.classifier(inputs = inputs,
                                      output_size = settings['output_size'],
                                      weights = settings['W'],
                                      biases = settings['b'],
                                      reg_scale = settings['reg'])

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=labels)

            correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(outputs, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            loss = tf.reduce_mean(cross_entropy)
            acc = 100*tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(outputs, 1), tf.argmax(labels, 1))))

        optimizer = tf.train.AdamOptimizer(learning_rate=settings['eta']).minimize(loss)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if settings['outputs'] == 'autoencoder':
                for epoch in range(settings['epochs']):
                    for batch_idx, batch in enumerate(batches):
                        feed = { inputs: batch }
                        #if(batch_idx % 30 == 0):
                        optimizer.run(feed_dict=feed)

                    cost = sess.run([loss], feed_dict={inputs: Y})
                    print('Epoch: ' + str(epoch) + ' Cost: ', cost)
                    loss_buff.append(cost)

                # extract weights
                W = tf.trainable_variables()[0]
                b = tf.trainable_variables()[1]
                W = sess.run(W.value())
                b = sess.run(b.value())
                sess.close()
            else:
                for epoch in range(settings['epochs']):
                    for idx, _ in enumerate(batches[0]):
                        feed = { inputs: batches[0][idx], labels: batches[1][idx] }
                        #if(batch_idx % 30 == 0):
                        sess.run([optimizer], feed_dict=feed)

                    cost = sess.run([loss], feed_dict={inputs: settings['train_X'], labels: settings['train_Y']})
                    t_cost = sess.run([loss], feed_dict={inputs: settings['test_X'], labels: settings['test_Y']})
                    print('Epoch: ' + str(epoch) + ' Cost: ', cost, 'test:', t_cost)
                    loss_buff.append(cost)
                    loss_test_buff.append(t_cost)
                finalAccuracy = sess.run(accuracy, feed_dict={inputs: settings['test_X'], labels: settings['test_Y']})
                print('Final accuracy:',finalAccuracy)
                return loss_buff, loss_test_buff

        # kill graph before iteration
        tf.reset_default_graph()
        return loss_buff, W, b


    def run(self):
        # setup data for build phase and train phase
        train_X = self.train_X
        train_Y_build = train_X.copy()
        train_Y = self.train_Y
        test_X = self.test_X
        test_Y_build = test_X.copy()
        test_Y = self.test_Y


        settings = {
            'BUILD_CONF': {
                'layers': [
                    {
                        'input_size': train_X.shape[1],
                        'X': train_X,
                        'Y': train_Y_build,
                        'hidden': 500,
                        'eta': 1e-3,
                        'reg': 1e-3, # 1e-3
                        'num_batches': 50,
                        'epochs': 100,
                        'outputs': 'autoencoder'
                    },
                    {
                        'input_size': None,
                        'X': None,
                        'Y': None,
                        'hidden': 120,
                        'eta': 1e-3,
                        'reg': 1e-3,
                        'num_batches': 20,
                        'epochs': 70,
                        'outputs': 'autoencoder'
                    }
                ],
                'save_weights': False,
            },
            'TEST_CONF': {
                'hidden': 'multilayer',
                'input_size': train_X.shape[1],
                'num_batches': 50,
                'epochs': 50,
                'eta': 7e-4,
                'reg': 1e-4,
                'plot_dim': (12,10),
                'plot_cost': False,
                'outputs': 'classifier',
                'test_X': test_X,
                'test_Y': test_Y,
                'train_X': train_X,
                'train_Y': train_Y,
                'output_size': 10,
                'W' : [],
                'b' : []
            }
        }


        # build phase
        for layer in settings['BUILD_CONF']['layers']:

            # setup data for next iteration
            if layer['input_size'] == None:
                assert len(settings['TEST_CONF']['W']) > 0
                layer['X'] = settings['TEST_CONF']['W'][-1]
                layer['Y'] = settings['TEST_CONF']['W'][-1]
                layer['input_size'] = layer['X'].shape[1]

            print(layer['X'].shape)
            save_path = self._config_to_path(layer)

            layer['batch_size'] = int(layer['input_size'] / layer['num_batches'])

            # make layer settings printable
            printable = layer.copy()
            del printable['X']
            del printable['Y']
            del printable['outputs']
            print('SETTINGS: ', json.dumps(printable, indent=2))

            # setup batches
            batches = np.array(np.array_split(layer['X'], layer['num_batches']))

            # train and store weights
            loss, W, b = self.train(layer, batches, layer['Y'])
            settings['TEST_CONF']['W'].append(W)
            settings['TEST_CONF']['b'].append(b)

            if settings['BUILD_CONF']['save_weights']:
                np.save('./data/__tfcache__/' + save_path + '_W', W)
                np.save('./data/__tfcache__/' + save_path + '_b', b)

            self.plot.loss(loss, label=['N=' + str(layer['hidden'])], save={'path': save_path+'_loss'})


        # testing phase

        print(settings['TEST_CONF']['train_X'].shape)
        print(settings['TEST_CONF']['train_Y'].shape)

        save_path = self._config_to_path(settings['TEST_CONF'])

        settings['TEST_CONF']['batch_size'] = int(settings['TEST_CONF']['input_size'] / settings['TEST_CONF']['num_batches'])

        batches = [np.array(np.array_split(settings['TEST_CONF']['train_X'], settings['TEST_CONF']['num_batches'])),
                   np.array(np.array_split(settings['TEST_CONF']['train_Y'], settings['TEST_CONF']['num_batches']))]

        # train and store weights
        train_loss, test_loss = self.train(settings['TEST_CONF'], batches, settings['TEST_CONF']['train_Y'])

        #print(test_loss)
