#!/usr/bin/python3.6
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import csv
from timeit import default_timer as timer
from sklearn.metrics import roc_auc_score, accuracy_score
from anomalyDetection import semisup_autoencoder, semisup_detection_inference

from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin

MAX_EVALS = 500

volume_dir = 'data/'
out_dir = '../out/'
#file_name = 'ae_test.csv'
#file_name_inference = 'ae_test_inference.csv'
file_name = 'train_caravan-insurance-challenge.csv'
file_name_inference = 'test_caravan-insurance-challenge.csv'
out_file = 'semisup_ae_trials.csv'
print('[TRAINING]Tot:5822	0:5474(94.02%)	1:348(5.98%)')  # [TRAINING]Tot:5822	0:5474(94.02%)	1:348(5.98%)
print('[TESTING]Tot:4000	0:3762(94.05%)	1:238(5.95%)')  # [TESTING]Tot:4000	0:3762(94.05%)	1:238(5.95%)

global ITERATION
ITERATION = 0

def data_distr():
    df_train = pd.read_csv('../data/{}'.format(file_name))
    df_test = pd.read_csv('../data/{}'.format(file_name_inference))

    # Extract the labels and format properly
    train_labels = np.array(df_train['label'].astype(np.int32)).reshape((-1,))
    test_labels = np.array(df_test['label'].astype(np.int32)).reshape((-1,))

    labels_train = train_labels[:]
    labels_test = test_labels[:]
    print('[TRAINING]Tot:{}\t0:{}({:.2f}%)\t1:{}({:.2f}%)'.format(len(list(labels_train)),list(labels_train).count(0),list(labels_train).count(0)/len(list(labels_train))*100,list(labels_train).count(1),list(labels_train).count(1)/len(list(labels_train))*100 ))
    print('[TESTING]Tot:{}\t0:{}({:.2f}%)\t1:{}({:.2f}%)'.format(len(list(labels_test)), list(labels_test).count(0),
                                                        list(labels_test).count(0) / len(list(labels_test)) * 100,
                                                        list(labels_test).count(1),
                                                        list(labels_test).count(1) / len(list(labels_test)) * 100))

    plt.figure()
    plt.hist(labels_train, edgecolor = 'k');
    plt.xlabel('Label'); plt.ylabel('Count'); plt.title('Counts of Training Labels');
    plt.show()

    plt.figure()
    plt.hist(labels_test, edgecolor = 'k');
    plt.xlabel('Label'); plt.ylabel('Count'); plt.title('Counts of Training Labels');
    plt.show()


def get_label(file_name):
    df = pd.read_csv('../data/{}'.format(file_name))
    # Extract the labels and format properly
    labels = np.array(df['label'].astype(np.int32)).reshape((-1,))
    return labels


def objective(params):
    """Objective function for SemiSup Autoencoder Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION
    ITERATION += 1

    # Conditional logic to assign top-level keys
    drop_factor = params['drop_enabled'].get('drop_factor', 0.1)

    # Extract the drop_enabled
    params['drop_enabled'] = params['drop_enabled']['drop_enabled']
    params['drop_factor'] = drop_factor
    # Extract overcomplete
    params['l1_reg'] = params['overcomplete']['l1_reg']
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
    if 'l1_reg' in params:
        params['l1_reg'] = round(params['l1_reg'], 5)


    start = timer()

    # Perform n_folds cross validation
    # cv_results = lgb.cv(params, train_set, num_boost_round=10000, nfold=n_folds,early_stopping_rounds=100, metrics='auc', seed=50)
    ae_model, scaler, ae_stats = semisup_autoencoder(volume_dir + file_name, sep=',', hparams_file=params, save=False)
    # for key, item in enumerate(ae_stats):
    #     print('key:{}   item:{}'.format(key, item))

    print('mae: {}'.format(ae_stats['MAE']))

    run_time = timer() - start

    # Extract the best score, since we are using the mse metric the best score is the minimum 'loss' value obtained
    best_score = np.min(ae_stats['val_loss'])

    print('best score: {}'.format(best_score))
    # Loss must be minimized
    loss = best_score

    # Boosting rounds that returned the highest cv score
    #n_estimators = int(np.argmax(cv_results['auc-mean']) + 1)

    # Write to the csv file ('a' means append)
    of_connection = open(out_dir+out_file, 'a', newline='')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, ITERATION, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            # 'estimators': n_estimators,
            'train_time': run_time, 'status': STATUS_OK}

space = {
        'epochs': 100,
        'batch_size': hp.quniform('batch_size', 8, 64, 8),
        'shuffle': hp.choice('shuffle', [True, False]),
        'overcomplete': hp.choice('overcomplete',
                                  [{'overcomplete': True, 'nl_o': hp.quniform('nl_o', 2, 10, 1),
                                    'nnl_o': hp.quniform('nnl_o', 5, 15, 1),
                                    'l1_reg': hp.quniform('l1_reg', 0.00001, 0.00003, 0.00001),
                                    'nl_u': 4,
                                    'nnl_u': 2},
                                   {'overcomplete': False, 'nl_o': 3,
                                    'nnl_o': 10,
                                    #'l1_reg': hp.quniform('l1_reg', 0.00001, 0.00003, 0.00001),
                                    'nl_u': hp.quniform('nl_u', 2, 10, 1),
                                    'nnl_u': hp.quniform('nnl_u', 2, 10, 1)}]),
        'actv': 'relu',
        'loss': 'mae',
        'lr': hp.loguniform('lr', np.log(0.01), np.log(0.2)),
        'optimizer': 'adam',
        'drop_enabled': hp.choice('drop_enabled',
                                  [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 1, 0.1)},
                                   {'drop_enabled': False, 'drop_factor': 0.1}])
    }
'''
space = {
    'class_weight': hp.choice('class_weight', [None, 'balanced']),
    'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)},
                                                 {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                                 {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 30, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0)
}'''
def get_params():
    space = {
        'epochs': 5,
        'batch_size': hp.quniform('batch_size', 8, 64, 8),
        'shuffle': hp.choice('shuffle', [True, False]),
        'overcomplete': hp.choice('overcomplete',
                                  [{'overcomplete': True, 'nl_o': hp.quniform('nl_o', 2, 10, 1),
                                    'nnl_o': hp.quniform('nnl_o', 5, 15, 1),
                                    'l1_reg': hp.quniform('l1_reg', 0.00001, 0.00003, 0.00001),
                                    'nl_u': 4,
                                    'nnl_u': 2},
                                   {'overcomplete': False, 'nl_o': 3,
                                    'nnl_o': 10,
                                    'l1_reg': hp.quniform('l1_reg', 0.00001, 0.00003, 0.00001),
                                    'nl_u': hp.quniform('nl_u', 2, 10, 1),
                                    'nnl_u': hp.quniform('nnl_u', 2, 10, 1)}]),
        'actv': 'relu',
        'loss': 'mae',
        'lr': hp.loguniform('lr', np.log(0.01), np.log(0.2)),
        'optimizer': 'adam',
        'drop_enabled': hp.choice('drop_enabled',
                                  [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 1, 0.1)},
                                   {'drop_enabled': False, 'drop_factor': 0.1}])
    }

    # Sample from the full space
    x = sample(space)

    # Conditional logic to assign top-level keys
    drop_factor = x['drop_enabled'].get('drop_factor', 0.1)

    # Extract the drop_enabled
    x['drop_enabled'] = x['drop_enabled']['drop_enabled']
    x['drop_factor'] = drop_factor
    # Extract overcomplete
    x['l1_reg'] = x['overcomplete']['l1_reg']
    x['nl_o'] = x['overcomplete']['nl_o']
    x['nnl_o'] = x['overcomplete']['nnl_o']
    x['nl_u'] = x['overcomplete']['nl_u']
    x['nnl_u'] = x['overcomplete']['nnl_u']
    x['overcomplete'] = x['overcomplete']['overcomplete']

    x['batch_size'] = int(x['batch_size'])
    x['nl_o'] = int(x['nl_o'])
    x['nnl_o'] = int(x['nnl_o'])
    x['nl_u'] = int(x['nl_u'])
    x['nnl_u'] = int(x['nnl_u'])
    x['drop_factor'] = round(x['drop_factor'], 2)
    x['l1_reg'] = round(x['l1_reg'], 5)
    return x

#data_distr() # VERY IMBALANCED DATA

# optimization algorithm
tpe_algorithm = tpe.suggest

# Keep track of results
bayes_trials = Trials()

# File to save first results
of_connection = open(out_dir+out_file, 'w', newline='')
writer = csv.writer(of_connection)
# Write the headers to the file
writer.writerow(['loss', 'params', 'iteration', 'train_time'])
of_connection.close()

#params = get_params()
#ret = objective(params)

# Run optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest,
            max_evals=MAX_EVALS, trials=bayes_trials, rstate=np.random.RandomState(50))

# Sort the trials with lowest loss (highest AUC) first
bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
print(bayes_trials_results[:10])

'''

model, scaler, sum = semisup_autoencoder(volume_dir+file_name, sep=',', hparams_file=params)
test_labels = get_label(file_name_inference)
predictions = semisup_detection_inference(volume_dir+file_name_inference, 'semisup_ae_default_00', sep=',')
auc = roc_auc_score(test_labels, predictions)
accuracy = accuracy_score(test_labels, predictions)

print('The baseline auc score on the test set is {:.4f}'.format(auc))
print('The baseline accuracy score on the test set is {:.4f}'.format(accuracy))

'''