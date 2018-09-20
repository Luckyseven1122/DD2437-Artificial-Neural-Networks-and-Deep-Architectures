import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print("Tensorflow version",tf.VERSION)
from decimal import *
from datagenerator import check_yes_no
import json
import re
import os.path

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

def save_settings(settings, path_name):
    del settings['inputs_dim']
    del settings['mg_time_series']
    print("Saving:", path_name)
    path_name = re.sub(r'[^\w\s\/-=]','',path_name)
    path_name = os.path.join('./tmp/', path_name + '.txt')
    with open(path_name, 'w+') as file:
        file.write(json.dumps(settings, indent=2, sort_keys=False)) # use `pickle.loads` to do the reverse
        file.close()

def plot_validation_noise(data, hidden_nodes, std, path):
    plt.close('all')
    for idx, nodes in enumerate(data):
        x = len(nodes)
        x = np.arange(0,x)
        print(idx)
        plt.plot(x, nodes, label=str(hidden_nodes[idx])+' hidden nodes')
    plt.legend()
    plt.ylabel('Validation MSE')
    plt.xlabel('Epochs')
    plt.savefig('./tmp/' + path + '_std='+ str(std) +'.png')

def plot_histograms(weights):
    # sort weights together
    opacity = [1, 0.8, 0.6]
    plt.ylabel('Frequency')
    for i, w in enumerate(weights):
        plt.subplot(1,3,i+1)
        for j, val in enumerate(w):
            plt.hist(val, bins=300, histtype='step', range=(-1.1, 1.1), label='W' + str(i+1))
        plt.legend()
    plt.show()

def plot_all(train_pred, valid_pred, test_pred, mg_time_series, path):
    plt.close('all')
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
    plt.ylabel('x(t)')
    plt.xlabel('t')
    plt.legend()
    #plt.show()
    plt.savefig('./tmp/' + path + '_all.png')
    #plt.pause(0.001)

def plot_time_series(x,path):
    plt.close('all')
    t = np.arange(0, x.shape[0])
    plt.plot(t, x, label='mg time series')
    plt.xlabel('t')
    plt.ylabel('x(t)')
    plt.legend()
    plt.savefig('./tmp/' + path + '_noisy_times_series.png')


def plot_predicted_vs_real(predicted, real):
    y = real
    plt.scatter(y, predicted)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--')
    plt.show()

def plot_prediction(prediction, test, path):
    plt.close('all')
    t = np.arange(0, test.shape[0])
    p = np.array(prediction)
    plt.plot(t, p, 'b', label='prediction')
    plt.plot(t, test, 'r', label='time series')
    plt.legend()
    #plt.show()
    plt.savefig('./tmp/' + path + '_prediction.png')

def plot_cost(train, valid, path):
    plt.close('all')
    x = np.arange(0, len(train))
    plt.plot(x, np.array(train), 'r', label='training cost')
    plt.plot(x, np.array(valid), 'b', label='validation cost')
    plt.xlabel('epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    #plt.show()
    plt.savefig('./tmp/' + path + '_cost.png')

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
    x = mg_time_series(t_stop+5) # add 5 for labels

    if std > 0:
        x += np.random.normal(0, std, x.shape)

    inputs = [x[t-20], x[t-15], x[t-10], x[t-5], x[t]]
    inputs = np.asarray(inputs)
    labels = x[t+5]

    # size of test according to task description
    test_size = 200
    validation_size = int(np.floor(inputs.shape[1] * validation_percentage))
    training_size = inputs.shape[1] - (test_size + validation_size)

    test_inputs  = inputs[:,training_size+validation_size:training_size+validation_size+test_size]
    test_labels  = labels[training_size+validation_size:training_size+validation_size+test_size]

    train_inputs = inputs[:,:training_size]
    train_labels = labels[:training_size]
    valid_inputs = inputs[:,training_size:training_size+validation_size]
    valid_labels = labels[training_size:training_size+validation_size]

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
    for idx, nodes in enumerate(settings['layers']):
        # first layer only
        if idx == 0:
            model.add(tf.keras.layers.Dense(nodes,
                                     input_dim=training['inputs'].shape[1],
                                     activation='tanh',
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l=settings['beta']) if settings['beta'] > 0 else None))
        else:
            model.add(tf.keras.layers.Dense(nodes,
                                     activation='tanh',
                                     kernel_initializer=tf.keras.initializers.he_normal(),
                                     kernel_regularizer=tf.keras.regularizers.l2(l=settings['beta']) if settings['beta'] > 0 else None))
    # add last layer
    model.add(tf.keras.layers.Dense(settings['outputs_dim']))

    model.compile(loss='mean_squared_error', optimizer='adam')


    #tf.keras.utils.plot_model(model, to_file='./tmp/model.png')

    # EarlyStopping
    if settings['min_delta'] > 0:
        callback = [tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=settings['min_delta'],
                                                     patience=settings['patience'])]
    else:
        callback = None

    # Training
    train_history = model.fit(training['inputs'], training['labels'],
                callbacks=callback,
                validation_data=(validation['inputs'], validation['labels']),
                batch_size=training['inputs'].shape[1],
                epochs=settings['epochs'])

    '''
    # extract weights
    weights = [w for w in model.trainable_weights if 'kernel' in w.name]
    _w = []
    for w in weights:
        _w.append(tf.keras.backend.get_value(w).flatten())
    return _w

    '''

    '''
    accuracy = model.evaluate(x=test['inputs'],
                              y=test['labels'],
                              batch_size=test['inputs'].shape[1])
    settings['accuracy'] = accuracy
    '''
    # Loss
    #t_loss = train_history.history['loss']
    #v_loss = train_history.history['val_loss']
    return train_history.history['val_loss']
    #plot_cost(t_loss, v_loss, settings['file_path'])


    # Store final loss
    #settings['training_final_mse'] = t_loss[-1]
    #settings['validation_final_mse'] = v_loss[-1]

    # Prediction
    #test_pred = model.predict(test['inputs'])
    #train_pred = model.predict(training['inputs'])
    #valid_pred = model.predict(validation['inputs'])
    #series = np.concatenate((np.concatenate((training['labels'], validation['labels'])), test['labels']))
    #plot_all(train_pred, valid_pred, test_pred, settings['mg_time_series'], settings['file_path'])
    #plot_prediction(test_pred, settings['mg_time_series'][-test_pred.size:], settings['file_path'])
    #plot_prediction(test_pred, test['labels'], settings['file_path'])

    #plot_time_series(settings['mg_time_series'], settings['file_path'])

    # Save config to file
    #save_settings(dict(settings.copy()), settings['file_path'])


    # get weights
    #outputs = [layer.get_weights() for layer in model.layers]
    #weights, biases = model.layers[0].get_weights()


def layer_to_str(layers):
    layer_path_name = ''
    for l in layers:
        layer_path_name += str(l)
    return layer_path_name

def task431():



    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=0)
    network_settings = {
        # [nr nodes in first hidden layer, ... , nr nodes in last hidden layer]
        'layers': [4, 3],
        'inputs_dim': int(training['inputs'].shape[1]),
        'outputs_dim': 1,
        'beta': 0,
        'mg_time_series': mg_time_series[300+5:1500+5],
        'epochs': 5,
        'eta': 0.00001,
        'patience': 8,
        'min_delta': 0.000000000000000008
    }

    network_settings['file_path'] = layer_to_str(network_settings['layers']) + \
                                                '_eta=' + str(network_settings['eta']) +\
                                                '_beta=' + str(network_settings['beta']) +\
                                                '_delta=' + str(network_settings['min_delta'])

    network(training, validation, test, network_settings)


def task432():


    training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=0)
    noise_std = [0.03, 0.09, 0.18]
    second_hidden_layer = [3, 6, 8]
    for idx, std in enumerate(noise_std):
        data = []
        for nodes in second_hidden_layer:
            training, validation, test, mg_time_series = generate_data(300, 1500, 0.3, std=std)

            network_settings = {
                # [nr nodes in first hidden layer, ... , nr nodes in last hidden layer]
                'layers': [9, nodes],
                'inputs_dim': int(training['inputs'].shape[1]),
                'outputs_dim': 1,
                'beta': 0, #10**(-5),
                'mg_time_series': mg_time_series[300+5:1500+5],
                'epochs': 100,
                'eta': 10**(-5),
                'patience': 8,
                'min_delta': 0#8*10**(-5)
            }

            network_settings['file_path'] = layer_to_str(network_settings['layers']) + \
                                                        '_std=' + str(std) +\
                                                        '_eta=' + str(network_settings['eta']) +\
                                                        '_beta=' + str(network_settings['beta']) +\
                                                        '_delta=' + str(network_settings['min_delta'])

            data.append(network(training, validation, test, network_settings))
        plot_validation_noise(data, second_hidden_layer, std, layer_to_str(network_settings['layers']))

#task431()
task432()
