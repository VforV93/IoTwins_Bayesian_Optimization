#!/usr/bin/python3.6
# import os
import numpy as np
import pandas as pd
import csv
from timeit import default_timer as timer
from anomalyDetection import _save_trained_model, semisup_autoencoder, semisup_detection_inference, sup_autoencoder_classr
import general_services as gs
from hyperopt import hp
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from functools import partial

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

volume_dir = '../data'
out_dir = '../out'


def load_trials(file_name):
    ret = gs.load_py_obj(file_name)
    print('[BO] trial loaded')
    return ret


def save_trials(trial, file_name):
    gs.serialize_py_obj(trial, file_name)
    print('[BO] trial saved')


def _bayesian_save_model(model_name, best_score, stats, model, scaler):
    print('_bayesian_save_model')
    return _save_trained_model(model_name, model, stats, scaler)


def get_label(file_name):
    df = pd.read_csv('../data/{}'.format(file_name))
    # Extract the labels and format properly
    labels = np.array(df['label'].astype(np.int32)).reshape((-1,))
    return labels


def iteration_generator(starting_value):
    while True:
        yield starting_value
        starting_value += 1


class Best_loss_in_run:
    def __init__(self, v=None):  # TODO accept a function/lamda function to confront the new value to update the current best_loss value
        self.best_loss = 1 if v is None else v

    def update(self, new_v):
        if new_v < self.best_loss:
            self.best_loss = new_v
            return True
        return False


def objective(params, function_to_optimize, trial_fname, iteration_gen, space_func_process=None, best_loss_threshold=None,
              others_params={}, save_model_func=_bayesian_save_model):
    """Objective function for SemiSup Autoencoder Hyperparameter Optimization"""
    iteration = next(iteration_gen)
    if best_loss_threshold is None:
        best_loss_threshold = Best_loss_in_run()

    print('best_loss_threshold: {}, iteration: {}'.format(best_loss_threshold.best_loss, iteration))
    # Conditional logic to assign top-level keys
    processed_params = space_func_process(params) if space_func_process is not None else params

    start = timer()

    # ae_model, scaler, ae_stats = function_to_optimize(file_name, hparams_file=params, n_percentile=n_percentile, **others_params)
    # ae_model, scaler, ae_stats = function_to_optimize(**processed_params, **others_params)
    best_score, stats, others_func_params = function_to_optimize(**processed_params, **others_params)

    run_time = timer() - start

    ''' '''
    # every best score obtained save the model
    if best_loss_threshold.update(best_score):
        print('new best loss score({}), saving model...'.format(round(best_score, 5)))
        model_name = 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
        # ae_model_fname, ae_stats_file, ae_scaler_file = _save_trained_model(gs.sanitize(ae_model_name), ae_model, ae_stats, scaler)
        save_params = save_model_func(gs.sanitize(model_name), best_score, stats, **others_func_params)

    # Loss must be minimized
    loss = best_score

    # Write to the csv file ('a' means append)
    of_connection = open('{}/{}'.format(out_dir, trial_fname), 'a', newline='')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, stats, iteration, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'stats': stats, 'iteration': iteration,
            'train_time': run_time, 'status': STATUS_OK}


def bayesian_optimization(function_to_optimize, trial_fname, space, space_func_process=None, opt_algorithm=None,
                          header_file=None, save_trial_every=None, total_evals=100, trials_name=None, others_params={},
                          save_model_func=_bayesian_save_model, user_id='default', task_id='0.0'):
    best_loss_threshold = None
    # optimization algorithm
    if opt_algorithm is None:
        opt_algorithm = tpe.suggest  # default value
    # header trials results file
    if header_file is None:
        header_file = ['loss', 'params', 'stats', 'iteration', 'train_time']  # default header  TODO senza stats forse è meglio come default
    # save Trials every tot step
    if save_trial_every is None or save_trial_every > total_evals:  # never saving trials while running the bayesian optimization
        save_trial_every = total_evals

    # Keep track of results
    if trials_name is None:
        bayes_trials = Trials()
        count_optimization = 0

        # create/overwrite the File to save first results
        of_connection = open('{}/{}'.format(out_dir, trial_fname), 'w', newline='')
        writer = csv.writer(of_connection)
        # Write the headers to the file
        writer.writerow(header_file)
        of_connection.close()
    else:
        bayes_trials = load_trials('{}/{}'.format(out_dir, trials_name))
        if bayes_trials is None:
            raise ValueError('Error in loading the trials({})'.format('{}/{}'.format(out_dir, trials_name)))
        count_optimization = len(bayes_trials.trials)
        best_loss_threshold = Best_loss_in_run(bayes_trials.best_trial['result']['loss'])

    # check if total_evals > count_optimization or not
    if total_evals <= count_optimization:
        raise ValueError('the number of eval evaluation specified(total_eval={}) '
                         'must be greater then the current value({})'.format(total_evals, count_optimization))

    # Run optimization
    fmin_objective = partial(objective, function_to_optimize=function_to_optimize, space_func_process=space_func_process,
                             trial_fname=trial_fname, iteration_gen=iteration_generator(count_optimization),
                             best_loss_threshold=best_loss_threshold, others_params=others_params, save_model_func=save_model_func)

    for i in range(int(count_optimization/save_trial_every)+1, int(total_evals/save_trial_every)+1):
        best = fmin(fn=fmin_objective, space=space, algo=opt_algorithm, max_evals=i*save_trial_every,
                    trials=bayes_trials, rstate=np.random.RandomState(50))

        # Trials saving
        save_trials(bayes_trials, '{}/trials_{}_{}_{}.p'.format(out_dir, i*save_trial_every, user_id, task_id))
        count_optimization += 1

    # Sort the trials with lowest loss first and print the best 10 ones
    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
    print(bayes_trials_results[:10])


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


def my_save_model(stats, **model_scaler):
    pass


s = {
        'epochs': 2,
        'batch_size': hp.quniform('batch_size', 8, 64, 8),
        'shuffle': hp.choice('shuffle', [True, False]),
        'overcomplete': hp.choice('overcomplete',
                                  [{'overcomplete': True, 'nl_o': hp.quniform('nl_o', 2, 10, 1),
                                    'nnl_o': hp.quniform('nnl_o', 5, 15, 1),
                                    'l1_reg': hp.quniform('l1_reg', 0.00001, 0.01, 0.0004995),
                                    'nl_u': 4, 'nnl_u': 2},
                                   {'overcomplete': False, 'nl_o': 3, 'nnl_o': 10,
                                    'nl_u': hp.quniform('nl_u', 2, 10, 1),
                                    'nnl_u': hp.quniform('nnl_u', 2, 10, 1)}]),
        'actv': 'relu',
        'loss': 'mae',
        'lr': hp.loguniform('lr', np.log(0.01), np.log(0.2)),
        'optimizer': 'adam',
        'drop_enabled': hp.choice('drop_enabled',
                                  [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 0.9, 0.1)},
                                   {'drop_enabled': False, 'drop_factor': 0.1}]),
        'n_percentile': hp.quniform('n_percentile', 40, 99, 1)
    }
df_n = 'train_caravan-insurance-challenge.csv'  # trial_fname
out_file = 'semisup_ae_trials.csv'              # trial_fname
s_t_e = None                                    # save_trial_every
t_e = 4                                         # total_evals
f_to_o = semisup_autoencoder_filter_stats       # function_to_optimize
o_p = {'df_fname': df_n, 'sep': ',', 'save': False,
       'user_id': 'default', 'task_id': '0.0'}  # others_params

bayesian_optimization(function_to_optimize=f_to_o, space_func_process=s_f_p, trial_fname=out_file, space=s,
                      save_trial_every=s_t_e, total_evals=t_e, others_params=o_p)  # , trials_name='trials_3_default_0.0.p')