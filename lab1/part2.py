import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print("Tensorflow version",tf.VERSION)

'''
    INSTALL TENSORFLOW:
    pip3 install -r requirements.txt
'''


def mg_time_series(t_stop):
    '''
    x(t+1) = x(t) + (0.2 * x(t-25))/(1 + x^10(t-25)) - 0.1x(t)

    x(0) = 1.5,
    x(t) = 0 for all t < 0
    '''
    beta = 0.2
    gamma = 0.1
    n = 10
    tau = 25

    # x(0)=1.5
    x = [1.5]
    for t in range(1, t_stop):
        delay = t - tau
        if delay < 0:
            delay = 0
        elif delay == 0:
            delay = 1.5
        else:
            delay = x[delay]
        x.append(x[-1] + ((0.2 * delay)/(1 + delay**n)) - 0.1*x[-1])
    return np.asarray(x)


def generate_data(t_start, t_stop, validation_percentage):
    t = np.arange(t_start, t_stop)
    x = mg_time_series(t_stop + 5) # add 5 for labels

    #plt.plot(t, x[t_start:])
    #plt.show()

    inputs = [x[t-20], x[t-15], x[t-10], x[t-5], x[t]]
    inputs = np.asarray(inputs)
    labels = x[t+5]

    # size of test according to task description
    test_size = 200
    validation_size = int(np.floor(inputs.shape[1] * validation_percentage))
    training_size = inputs.shape[1] - (test_size + validation_size)

    train_inputs = inputs[:,:training_size]
    train_labels = labels[:training_size]
    valid_inputs = inputs[:,training_size:training_size+validation_size]
    valid_labels = labels[training_size:training_size+validation_size]
    test_inputs  = inputs[:,training_size+validation_size:training_size+validation_size+test_size]
    test_labels  = labels[training_size+validation_size:training_size+validation_size+test_size]

    #training = tf.data.Dataset.from_tensors((tf.convert_to_tensor(train_inputs), tf.convert_to_tensor(train_labels)))
    #validation = tf.data.Dataset.from_tensors((tf.convert_to_tensor(valid_inputs), tf.convert_to_tensor(valid_labels)))
    #test = tf.data.Dataset.from_tensors((tf.convert_to_tensor(test_inputs), tf.convert_to_tensor(test_labels)))

    training = {'inputs': train_inputs.T, 'labels': train_labels.T}
    validation = {'inputs': valid_inputs.T, 'labels': valid_labels.T}
    test = {'inputs': test_inputs.T, 'labels': test_labels.T}

    return training, validation, test


def network(inputs, settings):
    assert len(settings['layers']) > 0
    assert settings['inputs_dim'] > 0
    assert settings['outputs_dim'] > 0
    assert settings['beta'] >= 0

    layers = []
    for idx, nodes in enumerate(settings['layers']):
        # first layer
        prev_nodes = settings['inputs_dim'] if idx == 0 else settings['layers'][idx-1]
        prev_input = inputs if idx == 0 else layers[-1]
        W = tf.Variable(tf.random_uniform([prev_nodes, nodes]))
        b = tf.Variable(tf.zeros([nodes]))
        layer = tf.add(tf.matmul(prev_input, W), b)
        # threshold function
        layers.append(tf.tanh(layer))

        if idx+1 == len(settings['layers']):
            W = tf.Variable(tf.random_uniform([settings['layers'][-1], settings['outputs_dim']]))
            b = tf.Variable(tf.zeros([settings['outputs_dim']]))
            output = tf.add(tf.matmul(layers[-1], W), b)
            # threshold ????
            # output = tf.tanh(output)
            weights = tf.trainable_variables()
            regularization = tf.add_n([ tf.nn.l2_loss(w) for w in weights if 'bias' not in w.name]) * settings['beta']
            return output, regularization


def train_network(training, validation, test, settings, prediction, optimizer, cost):
    cost_training = []
    cost_validation = []
    assert settings['epochs'] > 0
    print('Training starts')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(settings['epochs']):
            _, c_train = sess.run([optimizer, cost], feed_dict={inputs: training['inputs'], labels: training['labels']})
            cost_training.append(c_train)
            print('training cost:', c_train)


'''
EXECUTION STARTS HERE
'''

training, validation, test = generate_data(300, 1500, 0.2)

network_settings = {
    # [nr nodes in first hidden layer, ... , nr nodes in last hidden layer]
    'layers': [50, 5],
    'inputs_dim': int(training['inputs'].shape[1]),
    'outputs_dim': 1,
    'beta': 0
}

training_settings = {
    'epochs': 100,
    'eta': 0.001,
}

#inputs = tf.placeholder('float', training['inputs'].shape)
#labels = tf.placeholder('float', training['labels'].shape)

inputs = tf.placeholder('float')
labels = tf.placeholder('float')

prediction, regularization = network(inputs, network_settings)
cost = tf.reduce_mean(tf.square(prediction - labels) + regularization)
optimizer = tf.train.GradientDescentOptimizer(training_settings['eta']).minimize(cost)

train_network(training, validation, test, training_settings, prediction, optimizer, cost)
