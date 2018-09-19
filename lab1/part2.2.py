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

#plt.ion()
#plt.show()
inputs = tf.placeholder('float')
labels = tf.placeholder('float')

def save_file(path):
    if not check_yes_no(input("Store weights? Y/N: \n> ")):
        shutil.rmtree(path, ignore_errors=True)


def plot_all(train_pred, valid_pred, test_pred, mg_time_series):
    #plt.close('all')
    t_train = np.arange(0, train_pred.shape[0])
    t_valid = np.arange(train_pred.shape[0], train_pred.shape[0] + valid_pred.shape[0])
    t_test = np.arange( train_pred.shape[0] + valid_pred.shape[0],  train_pred.shape[0] + valid_pred.shape[0] + test_pred.shape[0])
    t = np.arange(0, train_pred.shape[0] + valid_pred.shape[0] + test_pred.shape[0])
    pred_line_x = train_pred.shape[0] + valid_pred.shape[0]
    pred_line_y_min = mg_time_series.min()
    pred_line_y_max = mg_time_series.max()
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
    #plt.pause(0.001)

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


def network(training, validation, test, settings):
    assert len(settings['layers']) > 0
    assert settings['inputs_dim'] > 0
    assert settings['outputs_dim'] > 0
    assert settings['beta'] >= 0

    layers = []
    model = tf.keras.Sequential()
    #initializer = tf.keras.initializers.he_normal()
    initializer = tf.keras.initializers.RandomNormal()
    for idx, nodes in enumerate(settings['layers']):
        # first layer only
        if idx == 0:
            model.add(tf.keras.layers.Dense(nodes,
                                     input_dim=training['inputs'].shape[1],
                                     activation='tanh',
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l=settings['beta'])))
        # last layers
        elif idx+1 == len(settings['layers']):
            model.add(tf.keras.layers.Dense(settings['outputs_dim']))
        else:
            model.add(tf.keras.layers.Dense(nodes,
                                     activation='tanh',
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l=settings['beta'])))

    model.compile(loss='mean_squared_error', optimizer='adam')

    callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                min_delta=settings['min_delta'],
                                                patience=settings['patience'])]

    # Training
    model.fit(training['inputs'], training['labels'],
                callbacks=callback,
                validation_data=(validation['inputs'], validation['labels']),
                batch_size=training['inputs'].shape[1],
                epochs=settings['epochs'])

    test_pred = model.predict(test['inputs'])
    train_pred = model.predict(training['inputs'])
    valid_pred = model.predict(validation['inputs'])
    plot_all(train_pred, valid_pred, test_pred, settings['mg_time_series'])




def layer_to_str(layers):
    layer_path_name = ''
    for l in layers:
        layer_path_name += str(l)
    return layer_path_name

def task431():
    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=0)
    network_settings = {
        # [nr nodes in first hidden layer, ... , nr nodes in last hidden layer]
        'layers': [7, 6],
        'inputs_dim': int(training['inputs'].shape[1]),
        'outputs_dim': 1,
        'beta': 0.001,
        'mg_time_series': mg_time_series[300:1500],
        'interactive': True,
        'epochs': 100,
        'eta': 0.00008,
        'patience': 8,
        'min_delta': 0.005,

    }


    network(training, validation, test, network_settings)

    #plot_time_series(mg_time_series)
    #plot_cost(cost_training, cost_validation)

    #plot_prediction(test_prediction, test['labels'])
    #plot_cost(cost_training, cost_validation)
    #plot_prediction(training_prediction, training['labels'])
    #plot_all(training_prediction, validation_prediction, test_prediction, training['labels'], validation['labels'], test['labels'])



def task432():

    std = [0.3, 0.09, 0.18]
    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=0.3)



task431()
#task432()
