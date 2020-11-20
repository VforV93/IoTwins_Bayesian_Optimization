import numpy as np
from services.bayesian_optimization.src.bayesianOptimization import sup_autoencoder_optimization
from hyperopt import hp

default_s = {
        'epochs': 150,
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

df_n = 'mammography.csv'  # dataset_fname
s_t_e = None              # save_trial_every
t_e = 50                  # total_evals

# save_model_func = None => No Keras model will never be saved during the optimization process
best, trial_fname = sup_autoencoder_optimization(df_n, default_s, total_evals=t_e, save_trial_every=s_t_e)
print("\n\ntrial_fname: {}".format(trial_fname))
print("BEST parameter/s:")
print(best)