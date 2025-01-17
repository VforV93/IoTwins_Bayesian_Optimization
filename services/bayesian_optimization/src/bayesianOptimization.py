#!/usr/bin/python3.6
import numpy as np
import csv
from timeit import default_timer as timer
from services.super_semisuper_anomaly_detection_services.src.anomalyDetection import _save_trained_model, \
    semisup_autoencoder, sup_autoencoder_classr
import services.super_semisuper_anomaly_detection_services.src.general_services as gs
from hyperopt import STATUS_OK
from hyperopt import tpe
from hyperopt import Trials

from hyperopt import fmin
from functools import partial
import os, sys

volume_dir = os.path.dirname(os.path.dirname(__file__))  # '../data'
out_dir = '{}/out'.format(volume_dir)  # '../out'
trained_models_dir = '{}/trained_models'.format(volume_dir)
# sys.setProperty("hadoop.home.dir", "c:\\winutil\\")

# internal function for loading previous saved Trials object(also a general .p object)
def _load_trials(file_name: str) -> object:
    """
    bayesianOptimization._load_trials::Load a Trials object already stored

    :params file_name : string
        the file name - only alphanumerical characters are allowed, no file
        extension needs to be specified

    :return : Trials object
        the Trials object
    """
    ret = gs.load_py_obj("{}.p".format(file_name))
    print('[BO] trial loaded')
    return ret


# internal function for storing Trials objects(also a general .p object)
def _save_trials(trial: object, file_name: str) -> None:
    """
    bayesianOptimization._save_trials::Serialize the Trials object

    :params trial : Python object
        Trials object to be serialized
    :params file_name : str
        name of the file of the serialized object
    """
    gs.serialize_py_obj(trial, file_name)
    print('[BO] trial saved')


# internal function for saving the Keras IoTwins models returned from the IoTwins services(semisup and sup)
def _bayesian_save_model(model_name: str, best_score: int, stats: dict, model, scaler):
    """
    bayesianOptimization._bayesian_save_model::default saving model function for applying the bayesian
    optimization to the IoTwins services.
    Save a Keras DL model already trained and stored, together with the stats computed during its
    validation and the sklearn object used for scaling the training data.

    :params model_name : string
        the model name - only alphanumerical characters are allowed, no file
        extension needs to be specified
    :params best_score : int
        the best score obtained from the objective function
    :params stats : Python dictionary
        the summary of the detailed statistics computed
    :params model : Keras model
        the trained model
    :params scaler : Sklearn scaler object
        the scaler used to normalize the data

    :return : string
        name of the file containing saved Keras model
    :return : string
        name of the pickle file containing Python dictionary with the summary
        of the detailed statistics
    :return : string
        name of the pickle file containing the scikit-learn object with scaler
        object used to normalize the data
    """
    print('[bayesianOptimization._bayesian_save_model] saving {} with score:{}'.format(model_name, best_score))
    return _save_trained_model(model_name, model, stats, scaler)


# internal generator for tracing the current number of the bayesian evaluation
def __iteration_generator(starting_value: int):
    """
    bayesianOptimization.__iteration_generator::Generator used to count properly
    the bayesian optimization evaluations, important for knowing the moment when
    saving the Trials object during the optimization process.

    :params starting_value : int
    the starting value of the iteration generator
    """
    while True:
        yield starting_value
        starting_value += 1


# internal Python Class for tracing the best current loss score obtained during the bayesian optimization
class Best_loss_in_run:
    """
    bayesianOptimization.Best_loss_in_run::Python class to store the best bayesian optimization loss.
    Best_loss_in_run.update:: compare a new value with the stored one.

        :params v : int
        starting best value stored

        :params compare_lf : lambda function
        function used if the 'new_compare_lf' parameter is None.
        This function is used to decide when update the best loss stored with the new
        value passed to the update method.

        :return : bool
            True if the new value is greater than the stored value, False otherwise.
    """

    def __init__(self, v=None, compare_lf=None):
        self.best_loss = 1 if v is None else v
        if compare_lf is None:
            self.compare_lf = lambda old, new: True if new < old else False
        else:
            self.compare_lf = compare_lf

    def update(self, new_v, new_compare_lf=None):
        comparing_values_lf = self.compare_lf if new_compare_lf is None else new_compare_lf
        if comparing_values_lf(self.best_loss, new_v):
            self.best_loss = new_v
            return True
        return False


# internal objective function that "wrap" the customer/client objective function
# the same parameters of the bayesian_optimization function
def __objective(params, function_to_optimize, trial_fname, iteration_gen, space_func_process=None,
                best_loss_threshold=None, others_params=None, save_model_func=_bayesian_save_model):
    if others_params is None:
        others_params = {}
    iteration = next(iteration_gen)

    # Conditional logic to assign top-level keys
    processed_params = space_func_process(params) if space_func_process is not None else params
    start = timer()

    best_score, stats, others_func_params = function_to_optimize(**processed_params, **others_params)

    run_time = timer() - start

    # every best score obtained save the model
    if save_model_func is not None and best_loss_threshold is not None and best_loss_threshold.update(best_score):
        print('new best loss score({}), saving model...'.format(round(best_score, 5)))
        model_name = 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
        save_params = save_model_func(gs.sanitize(model_name), best_score, stats, **others_func_params)

    # Loss must be minimized
    loss = best_score

    # Write to the csv file ('a' means append)
    of_connection = open('{}/{}'.format(out_dir, trial_fname), 'a', newline='')
    writer = csv.writer(of_connection)
    writer.writerow([loss, params, stats, iteration, run_time])

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': processed_params, 'stats': stats, 'iteration': iteration,
            'train_time': run_time, 'status': STATUS_OK}


# the service to be distributed to customers who want to apply the Bayesian Optimization
def bayesian_optimization(function_to_optimize, trial_fname, space, space_func_process=None, opt_algorithm=None,
                          save_model_func=_bayesian_save_model, save_trial_every=None, trials_name=None,
                          total_evals=100, others_params=None, user_id='default', task_id='0.0'):
    """
    bayesianOptimization.bayesian_optimization::probabilistic model based approach for finding the best
    hyperparameters configuration that guarantee to achieve the minimum of a given objective function that return
    a real-value metric keeping track of past evaluations.
    This function wrap the hyperopt open source python library: https://github.com/hyperopt/hyperopt

    :params function_to_optimize : function
        The objective function that must be minimized. It can be any function that
        returns a real value that we want to minimize. (If we want to maximize the
        returned real value, then we just have our function return the negative of that metric.)

        must return: score, stats, {...}
            - score: the return value of the function we want to minimize, this will
                     be passed to the save_model_func
            - stats: a Python dictionary containing all the objective function
                     performance information that we want to store in the trial_fname
                     csv file(e.g. accuracy, precision, recall etc.)
                     This will be passed to the save_model_func
            - {...}: a Python dictionary containing whatever we want, this will be
                     passed to the save_model_func (e.g. the model and the scaler)

    :params trial_fname : string
        name of the csv file which will be created to collect data(loss, parameters,
        objective func stats, iteration, training time) about the bayesian optimization process.

    :params space : Python dictionary
        the space over which to search. A search space consists of nested function
        expressions, including stochastic expressions. The stochastic expressions
        are the hyperparameters. The hyperparameter optimization algorithms work
        by replacing normal "sampling" logic with adaptive exploration strategies,
        which make no attempt to actually sample from the distributions specified
        in the search space. (see: Defining a Search Space in https://github.com/hyperopt/hyperopt/wiki/FMin)

    :params space_func_process : function, optional (the default is None).
    a function to process the "instance" of the space hyperparameters dict that receive as input.
    E.g. round off values, splitting the "main"/"big" instance to different dict
    to pass to the function_to_optimize, etc...

        must return: {...}
            - {...}: a Python dictionary containing whatever we want, this will
            be passed to the function_to_optimize as input

    :params opt_algorithm : function, optional (the default is the Tree of Parzen Estimators (TPE)).
    a implemented hyperopt algorithm between: Random Search, Tree of Parzen Estimators (TPE)-default value-, Adaptive TPE

    :params save_model_func : function, optional (the default is the _bayesian_save_model function).
    a function for saving the score, the stats and the others returned values
    from the objective function not only at the end of the bayesian optimization,
    but also every time a better hyperparameters configuration has been found.

        fixed input: file_name, best_score, stats, {...}
            - file_name: 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
            - best_score: the best score obtained in that run returned from the
                          function_to_optimize func
            - stats: a Python dictionary containing all the objective function
                     performance information returned from the function_to_optimize func
            - {...}: the optional Python dictionary returned from the function_to_optimize func

        return: void
        returned values/objects never used.

    :params save_trial_every : int, optional (the default is None).
    every how many iterations the Trials object file will be saved. See 'The Trials Object' in https://github.com/hyperopt/hyperopt/wiki/FMin
    Every 'save_trial_every' runs a new Trials_file.p will be created containing
    the information needed to apply the bayesian optimization.
    An existing file can be used to continue a previous run passing the file name
    to the 'trials_name' parameter.

    :params trials_name : string, optional (the default is None).
    the name of a previous Trials file saved locally. If not specified, a new bayesian optimization begins.
    Specifying a Trials file name the bayesian optimization process can continue from that 'checkpoint'.
    NB: if the Trials file contains a previous run of N iterations/evaluations,
    the 'total_evals' parameter must take into consideration of that and it has to be greater then N,
    otherwise the bayesian optimization will not begin/continue.
    E.g.    'total_evals' = 10
        after 10 evaluations the Trials object is going to be saved.
        If we want to perform more evaluations starting from the previous "results", it is necessary to specify
        the 'trials_name' and increment the 'total_evals'(>10)

    :params total_evals : int, optional (the default is 100).
    The number of iterations/evaluations the bayesian optimization will perform.

    :params others_params: Python dictionary, optional (the default is None).
    a Python dictionary containing whatever we want, this will be passed directly to the function_to_optimize without
    being processed by the space_func_process function.
    Use this dictionary for the fixed parameters necessary for the objective function.

    :params user_id : str, optional (the default is 'default').
        user identifier

    :params task_id : str, optional (the default is '0.0').
        task identifier


    :return : Python dictionary
        the best hyperparameters found according to the best loss score
    :return : string
        name of the pickle file containing the stored Trials object
    """

    # no fixed params to pass to the objective function
    if others_params is None:
        others_params = {}
    # optimization algorithm, default tpe
    if opt_algorithm is None:
        opt_algorithm = tpe.suggest  # default value
    # header trials results file
    header_file = ['loss', 'params', 'stats', 'iteration', 'train_time']  # default header(only choice)
    # save Trials every tot step/evaluations/iterations
    if save_trial_every is None or save_trial_every > total_evals:  # never saving trials while running the bayesian optimization
        save_trial_every = total_evals  # if not specified or is greater then the max number of evaluations
    if save_trial_every <= 0:
        raise ValueError('the number of evaluations after which to save the Trials object, save_trial_every({}) '
                         'must be strictly positive'.format(save_trial_every))

    # Keep track of the results
    if trials_name is None:  # starting from evaluation 0, no Trials object loaded
        bayes_trials = Trials()
        count_optimization = 0

        # create/overwrite the File(csv) to save the evaluations results(loss scores), parameters, stats and time
        of_connection = open('{}/{}'.format(out_dir, trial_fname), 'w', newline='')
        writer = csv.writer(of_connection)
        # Write the headers to the file
        writer.writerow(header_file)
        of_connection.close()
        best_loss_threshold = Best_loss_in_run()
    else:  # starting from a previous checkpoint, from a previous saved Trials object
        bayes_trials = _load_trials('{}/{}'.format(out_dir, trials_name))  # load the Trials object
        if bayes_trials is None:
            raise ValueError('Error in loading the trials({})'.format('{}/{}'.format(out_dir, trials_name)))
        count_optimization = len(bayes_trials.trials)  # check and store the previous number of evaluations
        best_loss_threshold = Best_loss_in_run(
            bayes_trials.best_trial['result']['loss'])  # check and store the previous best loss score

    # check if total_evals > count_optimization or not
    if total_evals <= count_optimization:
        raise ValueError('the number of eval evaluation specified(total_eval={}) '
                         'must be greater then the current value({})'.format(total_evals, count_optimization))

    # Run optimization
    fmin_objective = partial(__objective, function_to_optimize=function_to_optimize,
                             space_func_process=space_func_process,
                             trial_fname=trial_fname, iteration_gen=__iteration_generator(count_optimization),
                             best_loss_threshold=best_loss_threshold, others_params=others_params,
                             save_model_func=save_model_func)

    trial_fname = ""
    best = ""
    # loop for store locally the Trials object every 'save_trial_every' evaluations.
    # every 'save_trial_every' step, the Trials object is stored and continue with evaluations up to 'total_evals'
    for i in range(int(count_optimization / save_trial_every) + 1, int(total_evals / save_trial_every) + 1):
        best = fmin(fn=fmin_objective, space=space, algo=opt_algorithm, max_evals=i * save_trial_every,
                    trials=bayes_trials, rstate=np.random.RandomState(50))

        # Trials saving
        trial_fname = '{}/trials_{}_{}_{}.p'.format(out_dir, i * save_trial_every, user_id, task_id)
        _save_trials(bayes_trials, trial_fname)
        count_optimization += 1

    # Sort the trials with lowest loss first and print the best 10 ones
    # bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
    # print(bayes_trials_results[:10])

    return best, trial_fname


# The following services are wrappers for the semi-supervised and supervised services already available in IoTwins

# || --- Semisup_autoencoder Wrapper --- ||
def _semisup_s_f_p(params):
    """
    bayesianOptimization._semisup_s_f_p::the default space function process
    used in the semisup_autoencoder_optimization service.
    We will use the '_semisup_s_f_p' function to adjust/modify each given
    space instance and since the output is passed to the objective function
    we can round or cast values, we can also format the dictionary as the
    objective function expect it(no second level dictionary - just one level!,
    parameters assigned to the 'hparams_file' key, etc...).

    :params params : Python dictionary
        an instance of the domain space.

    :return : Python dictionary
        the dictionary that will be passed to the 'function_to_optimize'
        (_semisup_autoencoder_filter_stats) as input.
    """
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

    # 'n_percentile' not mandatory in the space dict, it can be a fixed parameter
    if 'n_percentile' in params:
        n_percentile = int(params['n_percentile'])
        ret_dict = {'hparams_file': params, 'n_percentile': n_percentile}
    else:
        ret_dict = {'hparams_file': params}

    return ret_dict


def _semisup_autoencoder_filter_stats(**params):
    """
    bayesianOptimization._semisup_autoencoder_filter_stats::Objective function for
    SemiSup Autoencoder Hyperparameter Optimization.
    This function "Wrap" the semisup_autoencoder service already developed in IoTwins.
    It takes in a processed/modified instance of the domain space that will be
    passed to the 'semisup_autoencoder' service.
    The output of the service is filtered before storing locally the stats.

    :params params : Python dictionary
        an instance of the domain space passed from the '_semisup_s_f_p' function

    :return : int
        the return value of the function we want to minimize, this will be passed
        to the save_model_func

    :return : Python dictionary
        a Python dictionary containing all the objective function performance
        information that we want to store in the trial_fname csv file
        (e.g. accuracy, precision, recall etc.).
        This will be passed to the save_model_func

    :return : Python dictionary
        a Python dictionary containing whatever we want, this will be passed
        to the save_model_func(e.g. the model and the scaler)
    """

    # the semisup_autoencoder service accept as input the name of the hparams file, not a dictionary.
    # So I save the hparams dictionary to a semisup_hparams.p file before passing the file name to the service.
    hparams_fname = '{}/semisup_hparams.p'.format(volume_dir)
    _save_trials(params['hparams_file'], hparams_fname)
    params['hparams_file'] = hparams_fname  # switch the dictionary object to the name of the file containing the same information
    model, scaler, stats = semisup_autoencoder(**params)

    # Filtering stats, the only stats we'll store in the csv file from the stats returned by the semisup_autoencoder IoTwins service
    key_stats = ['precision_N', 'recall_N', 'fscore_N', 'precision_A', 'recall_A', 'fscore_A', 'precision_W',
                 'recall_W', 'fscore_W', 'err_threshold', 'n_perc']
    filtered_stats = {}
    for i, (k, v) in enumerate(stats.items()):
        if k in key_stats:
            filtered_stats[k] = v

    score = model.history.history['val_loss'][-1]  # or score = 1 - np.max(stats['recall_A'])
    return score, filtered_stats, {'model': model, 'scaler': scaler}


#  (the default is -1, which implies summation over all axes).
def semisup_autoencoder_optimization(df_fname, space, sep=',', n_percentile=-1, trial_fname='bo_semisup_ae_trials.csv',
                                     space_func_process=_semisup_s_f_p, save_trial_every=None, save_model_func=-1,
                                     total_evals=100, trials_name=None, user_id='default', task_id='0.0',
                                     function_to_optimize=_semisup_autoencoder_filter_stats):
    """
    bayesianOptimization.semisup_autoencoder_optimization::probabilistic model based approach for finding the best
    hyperparameters configuration that guarantee to achieve the minimum of the semisup_autoencoder IoTwins service
    keeping track of past evaluations.
    This function wrap the service 'semisup_autoencoder' already developed in the IoTwins services.

    :params df_fname : str
        name of the csv file with data to be used for training and testing. One
        column has to termed "label" and it has to contains the class of the
        example - 0 means normal data point, any other integer number
        corresponds to an anomalous data point.

    :params space : Python dictionary
        the space over which to search. A search space consists of nested function
        expressions, including stochastic expressions. The stochastic expressions
        are the hyperparameters. The hyperparameter optimization algorithms work
        by replacing normal "sampling" logic with adaptive exploration strategies,
        which make no attempt to actually sample from the distributions specified
        in the search space. (see: Defining a Search Space in https://github.com/hyperopt/hyperopt/wiki/FMin)

    :params sep : string, optional (the default is ',').
        the columns separator used in the csv data file.

    :params n_percentile : int, optional
        percentile to be used to compute the detection threshold; if no
        percentile is provided all values in the range [85, 99] will be explored
        and the one providing the best accuracy results will be selected.

    :params trial_fname : string, optional (the default is 'bo_semisup_ae_trials.csv').
        name of the csv file which will be created to collect data(loss, parameters,
        objective func stats, iteration, training time) about the bayesian optimization process.

    :params space_func_process : function, optional (the default is the '_semisup_s_f_p' function).
        a function to process the "instance" of the space hyperparameters dict
        that receive as input. E.g. round off values, splitting the "main"/"big"
        instance to different dict to pass to the function_to_optimize, etc...

            must return: {...}
                - {...}: a Python dictionary containing whatever we want, this will
                         be passed to the function_to_optimize as input.

    :params save_trial_every : int, optional (the default is None).
        every how many iterations the Trials object file will be saved.
        See 'The Trials Object' in https://github.com/hyperopt/hyperopt/wiki/FMin
        Every 'save_trial_every' runs a new Trials_file.p will be created containing
        the information needed to apply the bayesian optimization.
        An existing file can be used to continue a previous run passing the file
        name to the 'trials_name' parameter.

    :params save_model_func : function, optional
        a function for saving the score, the stats and the others returned values
        from the objective function not only at the end of the bayesian optimization,
        but also every time a better hyperparameters configuration has been found.

            fixed input: file_name, best_score, stats, {...}
                - file_name: 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
                - best_score: the best score obtained in that run returned from the
                              function_to_optimize func
                - stats: a Python dictionary containing all the objective function
                         performance information returned from the function_to_optimize func
                - {...}: the optional Python dictionary returned from the function_to_optimize func

            return: void (The returned values/objects never be used)

    :params total_evals : int, optional (the default is 100).
        The number of iterations/evaluations the bayesian optimization will perform.

    :params trials_name : string, optional (the default is None).
        the name of a previous Trials file saved locally. If not specified, a new
        bayesian optimization begins. Specifying a Trials file name the bayesian
        optimization process can continue from that 'checkpoint'.
        NB: if the Trials file contains a previous run of N iterations/evaluations,
        the 'total_evals' parameter must take into consideration of that and it has
        to be greater then N otherwise the bayesian optimization will not begin/continue.
        E.g.    'total_evals' = 10
            after 10 evaluations the Trials object is going to be saved.
            If we want to perform more evaluations starting from the previous "results", it is necessary to specify
            the 'trials_name' and increment the 'total_evals'(>10)

    :params user_id : str, optional (the default is 'default').
        user identifier

    :params task_id : str, optional (the default is '0.0').
        task identifier

    :params function_to_optimize : function, optional (the default is the '_semisup_autoencoder_filter_stats' function).
        NB: This service has been meant to be a wrapper for the 'semisup_autoencoder'
        IoTwins service. Do not use this parameter to change the function to optimize
        but use it to change the score we want to minimize or to change the stats we
        want to keep track during the optimization process.
        To conclude, pass a function that uses/call the 'semisup_autoencoder' function
        (services.super_semisuper_anomaly_detection_services.semisup_autoencoder).

        If you want to optimize a generic function use the 'bayesian_optimization' service instead.

        must return: score, stats, {...}
            - score: the return value of the function we want to minimize, this will
                     be passed to the save_model_func
            - stats: a Python dictionary containing all the objective function performance
                     information that we want to store
                     in the trial_fname csv file(e.g. accuracy, precision, recall etc.)
                     This will be passed to the save_model_func
            - {...}: a Python dictionary containing whatever we want, this will be passed
                     to the save_model_func (e.g. the model and the scaler)

    :return : Python dictionary
        the best hyperparameters found according to the best loss score
    :return : string
        name of the pickle file containing the stored Trials object
    """

    # Preparing the fixed parameters for the bayesian_optimization function
    o_p = {'df_fname': df_fname, 'sep': sep, 'save': False,
           'user_id': user_id, 'task_id': task_id}  # others_params

    # if 'n_percentile' isn't directly passed to the function because it is contained in the space dictionary =>
    # I don't have to pass it as parameter to the 'bayesian_optimization' function through the 'others_params' dict
    if 'n_percentile' not in space:
        o_p['n_percentile'] = n_percentile

    s_m_f_kwargs = {
        'save_model_func': save_model_func}  # if -1 I won't pass the 'save_model_func' parameter to use the default save model function
    # otherwise, so if the user intentionally passed a custom function, I will pass that parameter to the bayesian_optimization
    # if 'save_model_func=None' I will pass that value to the bayesian_optimization and the models will never be saved!

    best, trial_fname = \
        bayesian_optimization(function_to_optimize=function_to_optimize, trial_fname=trial_fname,
                              space_func_process=space_func_process, space=space, save_trial_every=save_trial_every,
                              total_evals=total_evals, others_params=o_p, trials_name=trials_name,
                              **{k: v for k, v in s_m_f_kwargs.items() if v != -1})

    return best, trial_fname
# || --- --- --- --- --- --- --- --- --- --- ---  --- ||


# || --- Sup_autoencoder Wrapper --- ||

def _sup_s_f_p(params):
    """
    bayesianOptimization._sup_s_f_p::the default space function process
    used in the sup_autoencoder_optimization service.
    We will use the '_sup_s_f_p' function to adjust/modify each given
    space instance and since the output is passed to the objective
    function we can round or cast values, we can also format the
    dictionary as the objective function expect it(no second level dictionary
    - just one level!, parameters assigned to the 'hparams_file_ae' and
    'hparams_file_classr' keys, etc...).

    :params params : Python dictionary
    an instance of the domain space.

    :return : Python dictionary
        the dictionary that will be passed to the 'function_to_optimize'
        (_sup_autoencoder_filter_stats) as input.
    """
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


def _sup_autoencoder_filter_stats(**params):
    """
    bayesianOptimization._sup_autoencoder_filter_stats::Objective function for
    Sup Autoencoder Hyperparameter Optimization.
    This function "Wrap" the sup_autoencoder service already developed in IoTwins.
    It takes in a processed/modified instance of the domain space that will be
    passed to the 'semisup_autoencoder' service.
    The output of the service is filtered before storing locally the stats.

    :params params : Python dictionary
        an instance of the domain space passed from the '_sup_s_f_p' function

    :return : int
        the return value of the function we want to minimize, this will be passed
        to the save_model_func

    :return : Python dictionary
        a Python dictionary containing all the objective function performance
        information that we want to store in the trial_fname csv file
        (e.g. accuracy, precision, recall etc.).
        This will be passed to the save_model_func

    :return : Python dictionary
        a Python dictionary containing whatever we want, this will be passed
        to the save_model_func(e.g. the model and the scaler)
    """

    # the semisup_autoencoder service accept as input the name of the hparams file, not a dictionary.
    # So I save the hparams dictionary to a semisup_hparams.p file before passing the file name to the service.
    # hparams_file_ae   hparams_file_classr.
    hparams_ae_fname = '{}/sup_hparams_ae.p'.format(volume_dir)
    hparams_classr_fname = '{}/sup_hparams_classr.p'.format(volume_dir)
    _save_trials(params['hparams_file_ae'], hparams_ae_fname)
    _save_trials(params['hparams_file_classr'], hparams_classr_fname)
    params['hparams_file_ae'] = hparams_ae_fname  # switch the dictionary object to the name of the file containing the same information
    params['hparams_file_classr'] = hparams_classr_fname  # switch the dictionary object to the name of the file containing the same information
    model, scaler, stats = sup_autoencoder_classr(**params)
    score = model.history.history['val_loss'][-1]
    return score, stats, {'model': model, 'scaler': scaler}


def sup_autoencoder_optimization(df_fname, space, sep=',', trial_fname='bo_sup_ae_trials.csv',
                                 space_func_process=_sup_s_f_p, save_trial_every=None, save_model_func=-1,
                                 total_evals=100, trials_name=None, user_id='default', task_id='0.0',
                                 function_to_optimize=_sup_autoencoder_filter_stats):
    """
    bayesianOptimization.sup_autoencoder_optimization::probabilistic model based approach for finding the best
    hyperparameters configuration that guarantee to achieve the minimum of the sup_autoencoder IoTwins service
    keeping track of past evaluations.
    This function wrap the service 'sup_autoencoder' already developed in the IoTwins services.

    :params df_fname : str
        name of the csv file with data to be used for training and testing. One
        column has to termed "label" and it has to contains the class of the
        example - 0 means normal data point, any other integer number
        corresponds to an anomalous data point.

    :params space : Python dictionary
        the space over which to search. A search space consists of nested function
        expressions, including stochastic expressions. The stochastic expressions
        are the hyperparameters. The hyperparameter optimization algorithms work
        by replacing normal "sampling" logic with adaptive exploration strategies,
        which make no attempt to actually sample from the distributions specified
        in the search space. (see: Defining a Search Space in https://github.com/hyperopt/hyperopt/wiki/FMin)

    :params sep : string, optional (the default is ',').
        the columns separator used in the csv data file.

    :params trial_fname : string, optional (the default is 'bo_sup_ae_trials.csv').
        name of the csv file which will be created to collect data(loss, parameters,
        objective func stats, iteration, training time) about the bayesian optimization process.

    :params space_func_process : function, optional (the default is the '_sup_s_f_p' function).
        a function to process the "instance" of the space hyperparameters dict
        that receive as input. E.g. round off values, splitting the "main"/"big"
        instance to different dict to pass to the function_to_optimize, etc...

            must return: {...}
                - {...}: a Python dictionary containing whatever we want, this will
                         be passed to the function_to_optimize as input.

    :params save_trial_every : int, optional (the default is None).
        every how many iterations the Trials object file will be saved.
        See 'The Trials Object' in https://github.com/hyperopt/hyperopt/wiki/FMin
        Every 'save_trial_every' runs a new Trials_file.p will be created containing
        the information needed to apply the bayesian optimization.
        An existing file can be used to continue a previous run passing the file
        name to the 'trials_name' parameter.

    :params save_model_func : function, optional
        a function for saving the score, the stats and the others returned values
        from the objective function not only at the end of the bayesian optimization,
        but also every time a better hyperparameters configuration has been found.

            fixed input: file_name, best_score, stats, {...}
                - file_name: 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
                - best_score: the best score obtained in that run returned from the
                              function_to_optimize func
                - stats: a Python dictionary containing all the objective function
                         performance information returned from the function_to_optimize func
                - {...}: the optional Python dictionary returned from the function_to_optimize func

            return: void (The returned values/objects never be used)

    :params total_evals : int, optional (the default is 100).
        The number of iterations/evaluations the bayesian optimization will perform.

    :params trials_name : string, optional (the default is None).
        the name of a previous Trials file saved locally. If not specified, a new
        bayesian optimization begins. Specifying a Trials file name the bayesian
        optimization process can continue from that 'checkpoint'.
        NB: if the Trials file contains a previous run of N iterations/evaluations,
        the 'total_evals' parameter must take into consideration of that and it has
        to be greater then N otherwise the bayesian optimization will not begin/continue.
        E.g.    'total_evals' = 10
            after 10 evaluations the Trials object is going to be saved.
            If we want to perform more evaluations starting from the previous "results", it is necessary to specify
            the 'trials_name' and increment the 'total_evals'(>10)

    :params user_id : str, optional (the default is 'default').
        user identifier

    :params task_id : str, optional (the default is '0.0').
        task identifier

    :params function_to_optimize : function, optional (the default is the '_sup_autoencoder_filter_stats' function).
        NB: This service has been meant to be a wrapper for the 'sup_autoencoder'
        IoTwins service. Do not use this parameter to change the function to optimize
        but use it to change the score we want to minimize or to change the stats we
        want to keep track during the optimization process.
        To conclude, pass a function that uses/call the 'sup_autoencoder' function
        (services.super_semisuper_anomaly_detection_services.sup_autoencoder).

        If you want to optimize a generic function use the 'bayesian_optimization' service instead.

        must return: score, stats, {...}
            - score: the return value of the function we want to minimize, this will
                     be passed to the save_model_func
            - stats: a Python dictionary containing all the objective function performance
                     information that we want to store
                     in the trial_fname csv file(e.g. accuracy, precision, recall etc.)
                     This will be passed to the save_model_func
            - {...}: a Python dictionary containing whatever we want, this will be passed
                     to the save_model_func (e.g. the model and the scaler)

    :return : Python dictionary
        the best hyperparameters found according to the best loss score
    :return : string
        name of the pickle file containing the stored Trials object
    """
    # Preparing the fixed parameters for the bayesian_optimization function
    o_p = {'df_fname': df_fname, 'sep': sep, 'save': False,
           'user_id': user_id, 'task_id': task_id}  # others_params

    s_m_f_kwargs = {
        'save_model_func': save_model_func}  # if -1 I won't pass the 'save_model_func' parameter to use the default save model function
    # otherwise, so if the user intentionally passed a custom function, I will pass that parameter to the bayesian_optimization
    # if 'save_model_func=None' I will pass that value to the bayesian_optimization and the models will never be saved!

    best, trial_fname = \
        bayesian_optimization(function_to_optimize=function_to_optimize, trial_fname=trial_fname,
                              space_func_process=space_func_process, space=space, save_trial_every=save_trial_every,
                              total_evals=total_evals, others_params=o_p, trials_name=trials_name,
                              **{k: v for k, v in s_m_f_kwargs.items() if v != -1})

    return best, trial_fname
# || --- --- --- --- --- --- --- --- --- --- ---  --- ||