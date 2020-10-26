# import os
import numpy as np
from bayesianOptimization import bayesian_optimization
from anomalyDetection import semisup_autoencoder
from hyperopt import hp

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def s_f_p(params):
    drop_factor = params['drop_enabled'].get('drop_factor', 0.1)

    # Extract the drop_enabled
    params['drop_enabled'] = params['drop_enabled']['drop_enabled']
    params['drop_factor'] = drop_factor
    # Extract overcomplete

    if params['overcomplete']['overcomplete']:
        params['l1_reg'] = params['overcomplete']['l1_reg']
        params['l1_reg'] = round(params['l1_reg'], 5)

    params['nl_o'] = params['overcomplete']['nl_o']
    params['nnl_o'] = params['overcomplete']['nnl_o']
    params['nl_u'] = params['overcomplete']['nl_u']
    params['nnl_u'] = params['overcomplete']['nnl_u']
    params['overcomplete'] = params['overcomplete']['overcomplete']

    params['batch_size'] = int(params['batch_size'])
    params['nl_o'] = int(params['nl_o'])
    params['nnl_o'] = int(params['nnl_o'])
    params['nl_u'] = int(params['nl_u'])
    params['nnl_u'] = int(params['nnl_u'])
    params['drop_factor'] = round(params['drop_factor'], 2)

    n_percentile = int(params.pop('n_percentile'))
    return {'hparams_file': params, 'n_percentile': n_percentile}


def semisup_autoencoder_filter_stats(**params):
    model, scaler, stats = semisup_autoencoder(**params)

    # Filtering stats
    key_stats = ['precision_N', 'recall_N', 'fscore_N', 'precision_A', 'recall_A', 'fscore_A', 'precision_W',
                 'recall_W', 'fscore_W', 'err_threshold', 'n_perc']
    filtered_stats = {}
    for i, (k, v) in enumerate(stats.items()):
        if k in key_stats:
            filtered_stats[k] = v

    score = 1 - np.max(stats['recall_A'])
    return score, filtered_stats, {'model': model, 'scaler': scaler}


s = {
        'epochs': 50,
        'batch_size': hp.quniform('batch_size', 8, 64, 8),
        'shuffle': hp.choice('shuffle', [True, False]),
        'overcomplete': hp.choice('overcomplete',
                                  [{'overcomplete': True, 'nl_o': hp.quniform('nl_o', 1, 10, 1),
                                    'nnl_o': hp.quniform('nnl_o', 1, 15, 1),
                                    'l1_reg': hp.quniform('l1_reg', 0.00001, 0.01, 0.0004995),
                                    'nl_u': 4, 'nnl_u': 2},
                                   {'overcomplete': False, 'nl_o': 3, 'nnl_o': 10,
                                    'nl_u': hp.quniform('nl_u', 1, 10, 1),
                                    'nnl_u': hp.quniform('nnl_u', 1, 15, 1)}]),
        'actv': 'relu',
        'loss': 'mae',
        'lr': hp.loguniform('lr', np.log(0.001), np.log(0.02)),
        'optimizer': 'adam',
        'drop_enabled': hp.choice('drop_enabled',
                                  [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 0.9, 0.1)},
                                   {'drop_enabled': False, 'drop_factor': 0.1}]),
        'n_percentile': hp.quniform('n_percentile', 40, 99, 1)
    }

df_n = 'train_caravan-insurance-challenge.csv'  # dataset_fname
out_file = 'semisup_ae_trials.csv'              # trial_fname
s_t_e = None                                    # save_trial_every
t_e = 20                                        # total_evals
f_to_o = semisup_autoencoder_filter_stats       # function_to_optimize
o_p = {'df_fname': df_n, 'sep': ',', 'save': False,
       'user_id': 'default', 'task_id': '0.0'}  # others_params

bayesian_optimization(function_to_optimize=f_to_o, space_func_process=s_f_p, trial_fname=out_file, space=s,
                      save_trial_every=s_t_e, total_evals=t_e, others_params=o_p)  # , trials_name='trials_3_default_0.0.p')