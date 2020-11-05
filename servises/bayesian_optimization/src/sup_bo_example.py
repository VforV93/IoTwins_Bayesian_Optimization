# import os
import numpy as np
from bayesianOptimization import bayesian_optimization
from anomalyDetection import sup_autoencoder_classr
from hyperopt import hp

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def s_f_p(params):
    params_classr = {}
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

    params_classr['epochs'] = int(params.pop('epochs_classr_sup'))
    params_classr['batch_size'] = params['batch_size']
    params_classr['shuffle'] = params['shuffle']
    params_classr['nl'] = int(params.pop('nl_classr_sup'))
    params_classr['nnl'] = int(params.pop('nnl_classr_sup'))
    params_classr['actv'] = params.pop('actv_classr_sup')
    params_classr['loss'] = params.pop('loss_classr_sup')
    params_classr['lr'] = params.pop('lr_classr_sup')
    params_classr['optimizer'] = params.pop('optimizer_classr_sup')

    drop_factor = params['drop_enabled_classr_sup'].get('drop_factor_classr_sup', 0.1)

    # Extract the drop_enabled
    params_classr['drop_enabled'] = params.pop('drop_enabled_classr_sup')['drop_enabled_classr_sup']
    params_classr['drop_factor'] = drop_factor

    return {'hparams_file_ae': params, 'hparams_file_classr': params_classr}


def sup_autoencoder_filter_stats(**params):
    """Objective function for Sup Autoencoder+Classr Hyperparameter Optimization"""
    model, scaler, stats = sup_autoencoder_classr(**params)
    score = 1 - np.max(stats['recall'])
    return score, stats, {'model': model, 'scaler': scaler}


s = {
        'epochs': 1,
        'batch_size': hp.quniform('batch_size', 8, 64, 8),  # same value for autoencoder and classr
        'shuffle': hp.choice('shuffle', [True, False]),  # same value for autoencoder and classr
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
        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.002)),
        'optimizer': 'adam',
        'drop_enabled': hp.choice('drop_enabled',
                                  [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 0.9, 0.1)},
                                   {'drop_enabled': False, 'drop_factor': 0.1}]),

        'epochs_classr_sup': 1,
        'nl_classr_sup': hp.quniform('nl_classr_sup', 1, 10, 1),
        'nnl_classr_sup': hp.quniform('nnl_classr_sup', 1, 15, 1),
        'actv_classr_sup': 'relu',
        'loss_classr_sup': 'binary_crossentropy',
        'lr_classr_sup': hp.loguniform('lr_classr_sup', np.log(0.0001), np.log(0.002)),
        'optimizer_classr_sup': 'adam',
        'drop_enabled_classr_sup': hp.choice('drop_enabled_classr_sup',
                                  [{'drop_enabled_classr_sup': True, 'drop_factor_classr_sup': hp.quniform('drop_factor_classr_sup', 0.1, 0.9, 0.1)},
                                   {'drop_enabled_classr_sup': False, 'drop_factor_classr_sup': 0.1}])

    }

df_n = 'train_caravan-insurance-challenge.csv'  # dataset_fname
out_file = 'sup_ae_trials.csv'                  # trial_fname
s_t_e = None                                    # save_trial_every
t_e = 5                                         # total_evals
f_to_o = sup_autoencoder_filter_stats           # function_to_optimize
o_p = {'df_fname': df_n, 'sep': ',', 'save': False,
       'user_id': 'default', 'task_id': '0.0'}  # others_params

bayesian_optimization(function_to_optimize=f_to_o, space_func_process=s_f_p, trial_fname=out_file, space=s,
                      save_trial_every=s_t_e, total_evals=t_e, others_params=o_p)