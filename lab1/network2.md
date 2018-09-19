network_settings = {
    # [nr nodes in first hidden layer, ... , nr nodes in last hidden layer]
    'layers': [10,5],
    'inputs_dim': int(training['inputs'].shape[1]),
    'outputs_dim': 1,
    'beta': 0.00055,
}

layer_path_name = ''
for l in network_settings['layers']:
    layer_path_name += str(l)

training_settings = {
    'mg_time_series': mg_time_series[300:1500],
    'interactive': True,
    'epochs': 1000,
    'eta': 0.00001,
    'patience': 8,
    'min_delta': 0.0001,
    'weights_path': './tmp/' + str(network_settings['inputs_dim']) + \
                                       layer_path_name + \
                                       str(network_settings['outputs_dim']) + '_' + \
                                       str(network_settings['beta'])
}