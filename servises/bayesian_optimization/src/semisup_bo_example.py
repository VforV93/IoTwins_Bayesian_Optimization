import os, sys
import numpy as np
from servises.bayesian_optimization.src.bayesianOptimization import semisup_autoencoder_optimization
from hyperopt import hp

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

default_s = {
        'epochs': 100,
        'batch_size': hp.quniform('batch_size', 8, 64, 8),
        'shuffle': hp.choice('shuffle', [True, False]),
        'overcomplete': hp.choice('overcomplete',
                                  [{'overcomplete': True, 'nl_o': hp.quniform('nl_o', 1, 15, 1),
                                    'nnl_o': hp.quniform('nnl_o', 1, 15, 1),
                                    'l1_reg': hp.quniform('l1_reg', 0.00001, 0.01, 0.0004995),
                                    'nl_u': 4, 'nnl_u': 2},
                                   {'overcomplete': False, 'nl_o': 3, 'nnl_o': 10,
                                    'nl_u': hp.quniform('nl_u', 1, 10, 1),
                                    'nnl_u': hp.quniform('nnl_u', 1, 5, 1)}]),
        'actv': 'sigmoid',  # 'relu',
        'loss': 'binary_crossentropy',  # 'mae',
        'lr': hp.loguniform('lr', np.log(0.001), np.log(0.02)),
        'optimizer': 'adam',
        'drop_enabled': hp.choice('drop_enabled',
                                  [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 0.9, 0.1)},
                                   {'drop_enabled': False, 'drop_factor': 0.1}]),
        'n_percentile': hp.quniform('n_percentile', 40, 99, 1)  # not mandatory, if not expressed the best n_percentile threshold is searched
    }

df_n = 'mammography.csv'  # dataset_fname
s_t_e = None              # save_trial_every
t_e = 100                 # total_evals

# save_model_func = None => No Keras model will never be saved during the optimization process
semisup_autoencoder_optimization(df_n, default_s, save_model_func=None, total_evals=t_e, save_trial_every=s_t_e)

'''
bayesian_optimization(function_to_optimize=f_to_o, space_func_process=s_f_p, trial_fname=out_file, space=s,
                      save_trial_every=s_t_e, total_evals=t_e, others_params=o_p)  # , trials_name='trials_2_default_0.0')
'''