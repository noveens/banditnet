hyper_params = {
    'optimizer': 'SGD',
    'dataset': 'cifar',
    'num_sample': 1,
    'weight_decay': float(1e-5),
    'lr': 0.01,
    'epochs': 100,
    'batch_size': 128,
    'batch_log_interval': 50,
    'train_limit': int(500e3),
    'lamda': 0.9,
}

common_path  = hyper_params['dataset'] 
common_path += '_wd_' + str(hyper_params['weight_decay'])
common_path += '_lamda_' + str(hyper_params['lamda'])
common_path += '_train_limit_' + str(hyper_params['train_limit']) 

hyper_params['tensorboard_path'] = 'tensorboard_stuff/' + common_path
hyper_params['log_file'] = 'saved_logs/' + common_path
