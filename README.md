# IoTwins_Bayesian_Optimization
Filippo Lo Bue - Project of Work in Intelligent Systems

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Installation for Windows/Linux

```
Python Version 3.6
Anaconda 1.9.12
```

### Installing the environment from the yml file using <a href="https://www.anaconda.com/products/individual">Anaconda</a>

Open the **_conda prompt_** and navigate to the root folder of the project.\
Open the _IoTwins_env.yml_ file and replace '_<put here the %CONDA_PREFIX%>_' with the absolute path of your anaconda environments folder(_C:\Users\\$USER$\Anaconda3_), so you should have\
`prefix: C:\Users\$USER$\Anaconda3\envs\IoTwins` as last row.\
\
Create the conda environment from the yml file
```
(base) C:\Users\%USER%\IoTwins_Bayesian_Optimization> conda env create -f IoTwins_env.yml
```

You have just created the _IoTwins_ environment with all the packages in needed to run the examples and use the _bayesian_optimization_ service.

## Bayesian Optimization Service
The **_bayesian_optimization_** service developed in IoTwins is meant to support and allow clients to make hyperparameter fine-tuning.\
This service can be used to automatically improve the performances of the already existing services as well as future services that will be developed and added in the IoTwins context.\
Why was this service developed? The IoTwins services are applicable in across a wide range of contexts and specifically in the area of anomaly detection.\
To work best, these services need specific and not always easy to find configurations that usually involve experts' work. The _bayesian_optimization_ aims to remove this gap and allow any clients to automatically individuate the best hyperparameters configuration in an acceptable time.\
\
The _bayesian_optimization_ service is a wrapper of the <a href="https://github.com/hyperopt/hyperopt">Hyperopt</a> python library for serial and parallel optimization over awkward search spaces, which may include real-valued, discrete, and conditional dimensions.\
To conclude, the purpose of the service is to facilitate the use of the IoTwins services and make hyperparameter fine-tuning in the context of the client in the easiest and fastest manner.\
\
Formulating an optimization problem requires four main parts, two of which completly managed from the service:
- **Objective Function**[mandatory]: takes in an input and returns a loss to minimize.
- **Domain space**[mandatory]: the range of input values to evaluate.
- **Optimization Algorithm**[default tpe]: the method used to construct the surrogate function and choose the next values to evaluate. **Managed by the service. Use the default value!**
- **Results**[not mandatory]: score, value pairs that the algorithm uses to build the model. **Managed by the service.**
 
## How to use the Bayesian Optimization service
#### Example 1 - Basic example [Bayesian Optimization Service - Basic Example 1.ipynb]
How to optimize a polynomial function using the IoTwins optimization service.

#### Example 2 - the Rosenbrock function [Bayesian Optimization Service - Basic Example 2 - Rosenbrock function.ipynb]
How to optimize the Rosenbrock function using the IoTwins optimization service.

#### Example 3 - Iris Classifier [Bayesian Optimization Service - Example 3 - Machine Learning Model Example.ipynb]
How to optimize a Keras Classifier applied on the <a href="https://archive.ics.uci.edu/ml/datasets/iris">Iris</a> Dataset.

### The bayesianOptimization.bayesian_optimization service
Import the _bayesian_optimization_ service from 'services/bayesian_optimization/src/bayesianOptimization.py' and use it following the documentation below:
    
```python
services.bayesian_optimization.src.bayesianOptimization.bayesian_optimization(function_to_optimize, trial_fname, space, space_func_process=None, opt_algorithm=None,
                                                                              save_model_func=_bayesian_save_model, save_trial_every=None, trials_name=None,
                                                                              total_evals=100, others_params=None, user_id='default', task_id='0.0')
```
Probabilistic model based approach for finding the best hyperparameters configuration that guarantee to achieve the minimum of a given objective function that return a real-value metric keeping track of past evaluations.\
\
**Arguments**
- **function_to_optimize** : _function_\
The objective function that must be minimized. It can be any function that
returns a real value that we want to minimize. (If we want to maximize the
returned real value, we should have our function return the negative of that metric.)
    - must return: _score_, _stats_, _{...}_
        - **score**: the return value of the function we want to minimize, this will
                     be passed to the _save_model_func_
        - **stats**: a Python dictionary containing all the objective function
                     performance information that we want to store in the _trial_fname_
                     csv file(e.g. accuracy, precision, recall etc.).\
                     This will be passed to the _save_model_func_
        - **{...}**: a Python dictionary containing whatever we want, this will be
                     passed to the _save_model_func_ (e.g. the model and the scaler)
- **trial_fname** : _string_\
name of the csv file which will be created to collect data(loss, parameters,
objective func stats, iteration, training time) about the bayesian optimization process.
- **space** : _Python dictionary_\
the space over which to search. A search space consists of nested function
expressions, including stochastic expressions. The stochastic expressions
are the hyperparameters. The hyperparameter optimization algorithms work
by replacing normal "sampling" logic with adaptive exploration strategies,
which make no attempt to actually sample from the distributions specified
in the search space. (see: Defining a Search Space in https://github.com/hyperopt/hyperopt/wiki/FMin)
- **space_func_process** : _function_, optional (the default is None).\
a function to process the "instance" of the space hyperparameters dict that receive as input.\
E.g. round off values, splitting the "main"/"big" instance to different dict
to pass to the function_to_optimize, etc...
    - must return: _{...}_\
            - **{...}**: a Python dictionary containing whatever we want, this will
            be passed to the _function_to_optimize_ as input
- **opt_algorithm** : _function_, optional (the default is the **Tree of Parzen Estimators** (TPE)).\
a implemented hyperopt algorithm between: Random Search, Tree of Parzen Estimators (TPE)-default value-, Adaptive TPE
- **save_model_func** : _function_, optional (the default is the _\_bayesian_save_model_ function).\
a function for saving the score, the stats and the others returned values
from the objective function not only at the end of the bayesian optimization,
but also every time a better hyperparameters configuration has been found.
    - **fixed input**: _file_name_, _best_score_, _stats_, _{...}_
        - **file_name**: 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
        - **best_score**: the best score obtained in that run returned from the
                          _function_to_optimize_ func
        - **stats**: a Python dictionary containing all the objective function
                     performance information returned from the _function_to_optimize_ func
        - **{...}**: the optional Python dictionary returned from the _function_to_optimize_ func
    - **return**: _void_ (The returned values/objects never be used)
    
- **save_trial_every** : _int_, optional (the default is None).\
every how many iterations the Trials object file will be saved.\
See <a href="https://github.com/hyperopt/hyperopt/wiki/FMin">'The Trials Object'</a> in the official github page of Hyperopt.\
Every '_save_trial_every_' runs a new Trials_file.p will be created containing
the information needed to apply the bayesian optimization.
An existing file can be used to continue a previous run passing the file name
to the '_trials_name_' parameter.

- **trials_name** : _string_, optional (the default is None).\
the name of a previous Trials file saved locally. If not specified, a new bayesian optimization begins.
Specifying a Trials file name the bayesian optimization process can continue from that "checkpoint".\
NB: if the Trials file contains a previous run of N iterations/evaluations,
the '_total_evals_' parameter must take into consideration of that and it has to be greater then N,
otherwise the bayesian optimization will not begin/continue.\
E.g. &nbsp; 'total_evals' = 10\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;after 10 evaluations the Trials object is going to be saved.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If we want to perform more evaluations starting from the previous "results",\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;it is necessary to specify the 'trials_name' and increment the 'total_evals'(>10)
    
- **total_evals** : _int_, optional (the default is 100).\
The number of iterations/evaluations the bayesian optimization will perform.

- **others_params**: _Python dictionary_, optional (the default is None).\
a Python dictionary containing whatever we want, this will be passed directly to the _function_to_optimize_ without
being processed by the _space_func_process_ function.\
Use this dictionary for the fixed parameters necessary for the objective function.

- **user_id** : _string_, optional (the default is 'default').\
user identifier

- **task_id** : _string_, optional (the default is '0.0').
task identifier

**Return**
- **return** : _Python dictionary_\
the best hyperparameters found according to the best loss score

- **return** : _string_\
name of the pickle file containing the stored Trials object

### The bayesianOptimization.semisup_autoencoder_optimization service
Probabilistic model based approach for finding the best
hyperparameters configuration that guarantee to achieve 
the minimum of the **_semisup_autoencoder_** IoTwins service
keeping track of past evaluations.\
This function wrap the service '_semisup_autoencoder_' already developed in the IoTwins services.

Import the _semisup_autoencoder_optimization_ service from 'services/bayesian_optimization/src/bayesianOptimization.py' and use it following the documentation below:

```python
services.bayesian_optimization.src.bayesianOptimization.semisup_autoencoder_optimization(
                                        df_fname, space, sep=',', n_percentile=-1, trial_fname='bo_semisup_ae_trials.csv',
                                        space_func_process=_semisup_s_f_p, save_trial_every=None, save_model_func=-1,
                                        total_evals=100, trials_name=None, user_id='default', task_id='0.0',
                                        function_to_optimize=_semisup_autoencoder_filter_stats)
```
\
**Arguments**
- **df_fname** : _string_\
name of the csv file with data to be used for training and testing. One
column has to termed "label" and it has to contains the class of the
example - 0 means normal data point, any other integer number
corresponds to an anomalous data point.

- **space** : _Python dictionary_\
the space over which to search. A search space consists of nested function
expressions, including stochastic expressions. The stochastic expressions
are the hyperparameters. The hyperparameter optimization algorithms work
by replacing normal "sampling" logic with adaptive exploration strategies,
which make no attempt to actually sample from the distributions specified
in the search space. (see: Defining a Search Space in https://github.com/hyperopt/hyperopt/wiki/FMin)
       
- **sep** : _string_, optional (the default is ',')\
the columns separator used in the csv data file.

- **n_percentile** : _int_, optional\
percentile to be used to compute the detection threshold;\
if no percentile is provided all values in the range [85, 99] will be explored
and the one providing the best accuracy results will be selected.

- **trial_fname** : _string_, optional (the default is '_bo_semisup_ae_trials.csv_')\
name of the csv file which will be created to collect data(loss, parameters,
objective func stats, iteration, training time) about the bayesian optimization process.

- **space_func_process** : _function_, optional (the default is the '_\_semisup_s_f_p_' function)\
a function to process the "instance" of the space hyperparameters dict
that receive as input.\
E.g. round off values, splitting the "main"/"big"
instance to different dict to pass to the _function_to_optimize_, etc...

    - must return: _{...}_
        - **{...}**: a Python dictionary containing whatever we want, this will
                     be passed to the _function_to_optimize_ as input.

- **save_trial_every** : _int_, optional (the default is None)\
every how many iterations the Trials object file will be saved.
See <a href="https://github.com/hyperopt/hyperopt/wiki/FMin">'The Trials Object'</a> in the official github page of Hyperopt.\
Every '_save_trial_every_' runs a new Trials_file.p will be created containing
the information needed to apply the bayesian optimization.
An existing file can be used to continue a previous run passing the file
name to the '_trials_name_' parameter.

- **save_model_func** : _function_, optional\
a function for saving the score, the stats and the others returned values
from the objective function not only at the end of the bayesian optimization,
but also every time a better hyperparameters configuration has been found.
    - **fixed input**: _file_name_, _best_score_, _stats_, _{...}_
        - **file_name**: 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
        - **best_score**: the best score obtained in that run returned from the
                          _function_to_optimize_ func
        - **stats**: a Python dictionary containing all the objective function
                     performance information returned from the _function_to_optimize_ func
        - **{...}**: the optional Python dictionary returned from the _function_to_optimize_ func
    - **return**: _void_ (The returned values/objects never be used)

- **total_evals** : _int_, optional (the default is 100)\
The number of iterations/evaluations the bayesian optimization will perform.

- **trials_name** : _string_, optional (the default is None)\
the name of a previous Trials file saved locally. If not specified, a new bayesian optimization begins.
Specifying a Trials file name the bayesian optimization process can continue from that "checkpoint".\
NB: if the Trials file contains a previous run of N iterations/evaluations,
the '_total_evals_' parameter must take into consideration of that and it has to be greater then N,
otherwise the bayesian optimization will not begin/continue.\
E.g. &nbsp; 'total_evals' = 10\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;after 10 evaluations the Trials object is going to be saved.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If we want to perform more evaluations starting from the previous "results",\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;it is necessary to specify the 'trials_name' and increment the 'total_evals'(>10)

- **user_id** : _string_, optional (the default is 'default').\
user identifier

- **task_id** : _string_, optional (the default is '0.0').
task identifier

- **function_to_optimize** : _function_, optional (the default is the '_semisup_autoencoder_filter_stats' function)\
NB: This service has been meant to be a wrapper for the '_semisup_autoencoder_'
IoTwins service.\
Do not use this parameter to change the function to optimize
but use it to change the score we want to minimize or to change the stats we
want to keep track during the optimization process.\
To conclude, pass a function that uses/call the '_semisup_autoencoder_' function
(services.super_semisuper_anomaly_detection_services.semisup_autoencoder).\
\
If you want to optimize a generic function use the '_bayesian_optimization_' service instead.

    - must return: _score_, _stats_, _{...}_
        - **score**: the return value of the function we want to minimize, this will
                     be passed to the _save_model_func_
        - **stats**: a Python dictionary containing all the objective function performance
                     information that we want to store in the _trial_fname_ csv file(e.g. accuracy, precision, recall etc.)\
                     This will be passed to the _save_model_func_              
        - **{...}**: a Python dictionary containing whatever we want, this will be passed
                     to the _save_model_func_ (e.g. the model and the scaler)

**Return**
- **return** : _Python dictionary_
the best hyperparameters found according to the best loss score

- **return** : _string_
name of the pickle file containing the stored Trials object


###The bayesianOptimization.sup_autoencoder_optimization service
probabilistic model based approach for finding the best
hyperparameters configuration that guarantee to achieve the minimum of the sup_autoencoder IoTwins service
keeping track of past evaluations.\
This function wrap the service '_sup_autoencoder_' already developed in the IoTwins services.

Import the _sup_autoencoder_optimization_ service from 'services/bayesian_optimization/src/bayesianOptimization.py' and use it following the documentation below:

```python
services.bayesian_optimization.src.bayesianOptimization.sup_autoencoder_optimization(
                                        df_fname, space, sep=',', trial_fname='bo_sup_ae_trials.csv',
                                        space_func_process=_sup_s_f_p, save_trial_every=None, save_model_func=-1,
                                        total_evals=100, trials_name=None, user_id='default', task_id='0.0',
                                        function_to_optimize=_sup_autoencoder_filter_stats)
```
\
**Arguments**
- **df_fname** : _string_\
name of the csv file with data to be used for training and testing. One
column has to termed "label" and it has to contains the class of the
example - 0 means normal data point, any other integer number
corresponds to an anomalous data point.

- **space** : _Python dictionary_\
the space over which to search. A search space consists of nested function
expressions, including stochastic expressions. The stochastic expressions
are the hyperparameters. The hyperparameter optimization algorithms work
by replacing normal "sampling" logic with adaptive exploration strategies,
which make no attempt to actually sample from the distributions specified
in the search space. (see: Defining a Search Space in https://github.com/hyperopt/hyperopt/wiki/FMin)
       
- **sep** : _string_, optional (the default is ',')\
the columns separator used in the csv data file.

- **trial_fname** : _string_, optional (the default is 'bo_sup_ae_trials.csv')\
name of the csv file which will be created to collect data(loss, parameters,
objective func stats, iteration, training time) about the bayesian optimization process.

- **space_func_process** : _function_, optional (the default is the '_\_sup_s_f_p_' function)\
a function to process the "instance" of the space hyperparameters dict
that receive as input.\
E.g. round off values, splitting the "main"/"big"
instance to different dict to pass to the _function_to_optimize_, etc...

    - must return: _{...}_
        - **{...}**: a Python dictionary containing whatever we want, this will
                     be passed to the _function_to_optimize_ as input.

- **save_trial_every** : _int_, optional (the default is None)\
every how many iterations the Trials object file will be saved.
See <a href="https://github.com/hyperopt/hyperopt/wiki/FMin">'The Trials Object'</a> in the official github page of Hyperopt.\
Every '_save_trial_every_' runs a new Trials_file.p will be created containing
the information needed to apply the bayesian optimization.
An existing file can be used to continue a previous run passing the file
name to the '_trials_name_' parameter.

- **save_model_func** : _function_, optional\
a function for saving the score, the stats and the others returned values
from the objective function not only at the end of the bayesian optimization,
but also every time a better hyperparameters configuration has been found.
    - **fixed input**: _file_name_, _best_score_, _stats_, _{...}_\
        - **file_name**: 'bayesian_opt_model(score_{})'.format(round(best_score, 5))
        - **best_score**: the best score obtained in that run returned from the
                          _function_to_optimize_ func
        - **stats**: a Python dictionary containing all the objective function
                     performance information returned from the _function_to_optimize_ func
        - **{...}**: the optional Python dictionary returned from the _function_to_optimize_ func
    - **return**: _void_ (The returned values/objects never be used)

- **total_evals** : _int_, optional (the default is 100)\
The number of iterations/evaluations the bayesian optimization will perform.

- **trials_name** : _string_, optional (the default is None)\
the name of a previous Trials file saved locally. If not specified, a new bayesian optimization begins.
Specifying a Trials file name the bayesian optimization process can continue from that "checkpoint".\
NB: if the Trials file contains a previous run of N iterations/evaluations,
the '_total_evals_' parameter must take into consideration of that and it has to be greater then N,
otherwise the bayesian optimization will not begin/continue.\
E.g. &nbsp; 'total_evals' = 10\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;after 10 evaluations the Trials object is going to be saved.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If we want to perform more evaluations starting from the previous "results",\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;it is necessary to specify the 'trials_name' and increment the 'total_evals'(>10)

- **user_id** : _string_, optional (the default is 'default').\
user identifier

- **task_id** : _string_, optional (the default is '0.0').
task identifier

- **function_to_optimize** : _function_, optional (the default is the '_sup_autoencoder_filter_stats' function)\
NB: This service has been meant to be a wrapper for the '_sup_autoencoder_'
IoTwins service.\
Do not use this parameter to change the function to optimize
but use it to change the score we want to minimize or to change the stats we
want to keep track during the optimization process.\
To conclude, pass a function that uses/call the '_sup_autoencoder_' function
(services.super_semisuper_anomaly_detection_services.sup_autoencoder).\
\
If you want to optimize a generic function use the '_bayesian_optimization_' service instead.

    - must return: _score_, _stats_, _{...}_
        - **score**: the return value of the function we want to minimize, this will
                     be passed to the _save_model_func_
        - **stats**: a Python dictionary containing all the objective function performance
                     information that we want to store in the _trial_fname_ csv file(e.g. accuracy, precision, recall etc.)\
                     This will be passed to the _save_model_func_              
        - **{...}**: a Python dictionary containing whatever we want, this will be passed
                     to the _save_model_func_ (e.g. the model and the scaler)


**Return**
- **return** : _Python dictionary_
the best hyperparameters found according to the best loss score

- **return** : _string_
name of the pickle file containing the stored Trials object

### References
- [Hyperopt Github Page](https://github.com/hyperopt/hyperopt)
- [A Conceptual Explanation of Bayesian Hyperparameter Optimization for Machine Learning](https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f)
- [An Introductory Example of Bayesian Optimization in Python with Hyperopt](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0)

## Author
* **Filippo Lo Bue**