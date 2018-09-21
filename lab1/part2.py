import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print("Tensorflow version",tf.VERSION)
from decimal import *
from datagenerator import check_yes_no


'''
    INSTALL TENSORFLOW:
    pip3 install -r requirements.txt
'''

plt.ion()
plt.show()
inputs = tf.placeholder('float')
labels = tf.placeholder('float')

def save_file(path):
    if not check_yes_no(input("Store weights? Y/N: \n> ")):
        shutil.rmtree(path, ignore_errors=True)


def plot_all(train_pred, valid_pred, test_pred, train, valid, test, mg_time_series):
    plt.close('all')
    t_train = np.arange(0, train.shape[0])
    t_valid = np.arange(train.shape[0], train.shape[0] + valid.shape[0])
    t_test = np.arange( train.shape[0] + valid.shape[0],  train.shape[0] + valid.shape[0] + test.shape[0])
    t = np.arange(0, train.shape[0] + valid.shape[0] + test.shape[0])
    pred_line_x = train.shape[0] + valid.shape[0]
    pred_line_y_min = test.min()
    pred_line_y_max = test.max()
    #print(t_train.shape, t_valid.shape, t_test.shape)
    opacity = 0.5
    plt.figure(figsize=(15,6))
    plt.axis((0,1200,0,1.6))
    plt.plot(t, mg_time_series,'g', alpha=opacity, label='MG time series')
    #plt.plot(t_train, train, 'r', alpha=opacity)
    plt.plot(np.concatenate((t_train, t_valid)), np.concatenate((train_pred, valid_pred)), 'b', label='Trainig/Validation learning')
    #plt.plot(t_valid, valid, 'r', alpha=opacity)
    #plt.plot(t_valid, valid_pred, 'b', label='Trainig/Validation data')
    #plt.plot(t_test, test, 'r', alpha=opacity)
    plt.plot(t_test, test_pred, '--b', label='Prediction')
    plt.plot([pred_line_x, pred_line_x], [pred_line_y_min, pred_line_y_max], '--k')
    plt.legend()
    plt.show()
    plt.pause(0.001)

def plot_time_series(x):
    t = np.arange(0, x.shape[0])
    plt.plot(t, x)
    plt.show()

def plot_predicted_vs_real(predicted, real):
    y = real
    plt.scatter(y, predicted)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--')
    plt.show()

def plot_prediction(prediction, test):
    t = np.arange(0, test.shape[0])
    p = np.array(prediction)
    plt.plot(t, p, 'b')
    plt.plot(t, test, 'r')
    plt.show()

def plot_cost(train, valid):
    x = np.arange(0, len(train))
    plt.plot(x, np.array(train), 'r', label='training cost')
    plt.plot(x, np.array(valid), 'b', label='validation cost')
    plt.xlabel('epochs', fontsize=12)
    plt.legend()
    plt.show()

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
        x.append(x[-1] + ((beta * delay)/(1 + delay**n)) - gamma*x[-1])
    return np.asarray(x)


def generate_data(t_start, t_stop, validation_percentage, std):
    t = np.arange(t_start, t_stop)
    x = mg_time_series(t_stop + 5) # add 5 for labels

    if std > 0:
        x += np.random.normal(0, std, x.shape)

    #plt.plot(t, x[t_start:])
    #plt.show()

    inputs = [x[t-20], x[t-15], x[t-10], x[t-5], x[t]]
    inputs = np.asarray(inputs)
    labels = x[t+5]


    # does any difference?
    #idx = np.random.permutation(inputs.shape[1])
    #inputs = inputs[:,idx]
    #labels = labels[idx]

    # size of test according to task description
    test_size = 200
    validation_size = int(np.floor(inputs.shape[1] * validation_percentage))
    training_size = inputs.shape[1] - (test_size + validation_size)

    test_inputs  = inputs[:,training_size+validation_size:training_size+validation_size+test_size]
    test_labels  = labels[training_size+validation_size:training_size+validation_size+test_size]

    #idx = np.random.permutation(training_size+validation_size)
    #inputs = inputs[:,idx]
    #labels = labels[idx]

    train_inputs = inputs[:,:training_size]
    train_labels = labels[:training_size]
    valid_inputs = inputs[:,training_size:training_size+validation_size]
    valid_labels = labels[training_size:training_size+validation_size]


    #training = tf.data.Dataset.from_tensors((tf.convert_to_tensor(train_inputs), tf.convert_to_tensor(train_labels)))
    #validation = tf.data.Dataset.from_tensors((tf.convert_to_tensor(valid_inputs), tf.convert_to_tensor(valid_labels)))
    #test = tf.data.Dataset.from_tensors((tf.convert_to_tensor(test_inputs), tf.convert_to_tensor(test_labels)))

    training = {'inputs': train_inputs.T, 'labels': train_labels.T}
    validation = {'inputs': valid_inputs.T, 'labels': valid_labels.T}
    test = {'inputs': test_inputs.T, 'labels': test_labels.T}

    return training, validation, test, x


def network(inputs, settings):
    assert len(settings['layers']) > 0
    assert settings['inputs_dim'] > 0
    assert settings['outputs_dim'] > 0
    assert settings['beta'] >= 0

    layers = []
    #initializer = tf.keras.initializers.he_normal()
    initializer = tf.keras.initializers.RandomNormal()
    for idx, nodes in enumerate(settings['layers']):
        # first layer
        prev_nodes = settings['inputs_dim'] if idx == 0 else settings['layers'][idx-1]
        prev_input = inputs if idx == 0 else layers[-1]
        W = tf.Variable(initializer([prev_nodes, nodes]), name='weight')
        b = tf.Variable(tf.zeros([nodes]), name='bias')
        layer = tf.add(tf.matmul(prev_input, W), b)
        layer = tf.matmul(prev_input, W)
        layers.append(tf.nn.tanh(layer))

        if idx+1 == len(settings['layers']):
            W = tf.Variable(initializer([settings['layers'][-1], settings['outputs_dim']]), name='weight')
            b = tf.Variable(tf.zeros([settings['outputs_dim']]), name='bias')
            output = tf.add(tf.matmul(layers[-1], W))
            output = tf.matmul(layers[-1], W)
            weights = tf.trainable_variables()
            regularization = tf.add_n([ tf.nn.l2_loss(w) for w in weights if 'bias' not in w.name]) * settings['beta']
            return output, regularization



def train_network(training, validation, test, settings, prediction, optimizer, cost):
    cost_training = []
    cost_validation = []
    std = []
    assert settings['epochs'] > 0
    assert settings['patience'] > 0
    assert settings['min_delta'] > 0
    assert isinstance(settings['interactive'], bool)
    print('Training starts')

    saver = tf.train.Saver()

    patience_counter = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter("./tmp/log", sess.graph)

        # Try loading weights
        try:
            print("Trying to load weights:",settings['weights_path'] + '/data.ckpt')
            saver.restore(sess=sess, save_path=settings['weights_path'] + '/data.ckpt') #"/tmp/model.ckpt"
            print("Success!")
        except:
            print("Error. Storing current weights.")
            saver.save(sess=sess, save_path=settings['weights_path'] + '/data.ckpt')

        for e in range(settings['epochs']):
            _, c_train = sess.run([optimizer, cost], feed_dict={inputs: training['inputs'], labels: training['labels']})
            c_valid = sess.run(cost, feed_dict={inputs: validation['inputs'], labels: validation['labels']})
            cost_training.append(c_train)
            cost_validation.append(c_valid)
            print('training cost:', c_train, 'valid cost:', c_valid, "Delta validation:", cost_validation[e-1] - cost_validation[e])


            if settings['interactive'] and (e % 5) == 0:
                test_prediction = sess.run(prediction, feed_dict={inputs: test['inputs']})
                training_prediction = sess.run(prediction, feed_dict={inputs: training['inputs']})
                validation_prediction = sess.run(prediction, feed_dict={inputs: validation['inputs']})
                plot_all(training_prediction, validation_prediction, test_prediction, training['labels'], validation['labels'], test['labels'], settings['mg_time_series'])



            if e > 0 and (cost_validation[e-1] - cost_validation[e]) > settings['min_delta']:
                patience_counter = 0
            else:
                print(e, "hej")
                patience_counter += 1

            if patience_counter > settings['patience']:
                print("early stopping...")
                break

        c_test = sess.run(cost, feed_dict={inputs: test['inputs'], labels: test['labels']})
        print('Test:', c_test)
        test_prediction = sess.run(prediction, feed_dict={inputs: test['inputs']})
        training_prediction = sess.run(prediction, feed_dict={inputs: training['inputs']})
        validation_prediction = sess.run(prediction, feed_dict={inputs: validation['inputs']})

        writer.close()

    return cost_training, cost_validation, test_prediction, training_prediction, validation_prediction


def task431():
    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=0)
    network_settings = {
        # [nr nodes in first hidden layer, ... , nr nodes in last hidden layer]
        'layers': [7, 6],
        'inputs_dim': int(training['inputs'].shape[1]),
        'outputs_dim': 1,
        'beta': 0.0000001,
    }

    layer_path_name = ''
    for l in network_settings['layers']:
        layer_path_name += str(l)

    training_settings = {
        'mg_time_series': mg_time_series[300:1500],
        'interactive': True,
        'epochs': 1000,
        'eta': 0.0008,
        'patience': 8,
        'min_delta': 0.00000005,
        'weights_path': './tmp/' + str(network_settings['inputs_dim']) + \
                                           layer_path_name + \
                                           str(network_settings['outputs_dim']) + '_' + \
                                           str(network_settings['beta'])
    }


    prediction, regularization = network(inputs, network_settings)
    cost = tf.reduce_mean(tf.square(prediction - labels) + tf.sqrt(regularization))

    optimizer = tf.train.AdamOptimizer(learning_rate=training_settings['eta']).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(training_settings['eta']).minimize(cost)
    cost_training, cost_validation, test_prediction, training_prediction, validation_prediction = train_network(training, validation, test, training_settings, prediction, optimizer, cost)

    #plot_time_series(mg_time_series)
    #plot_cost(cost_training, cost_validation)
    plt.close('all')
    plt.ioff()
    #plot_prediction(test_prediction, test['labels'])
    #plot_cost(cost_training, cost_validation)
    #plot_prediction(training_prediction, training['labels'])
    #plot_all(training_prediction, validation_prediction, test_prediction, training['labels'], validation['labels'], test['labels'])

    save_file(training_settings['weights_path'])


def task432():

    std = [0.3, 0.09, 0.18]
    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=0.3)



task431()
#task432()
