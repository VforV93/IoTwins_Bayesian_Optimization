import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import precision_recall_fscore_support

from hyperopt import hp

from servises.bayesian_optimization.src.bayesianOptimization import bayesian_optimization, trained_models_dir  # import the service

# ||--- Data Preperation ---||
iris = load_iris()
X = iris['data']
y = iris['target']
names = iris['target_names']
feature_names = iris['feature_names']

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

# Scale data to have mean 0 and variance 1
# which is importance for convergence of the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data set into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=41)

n_features = X.shape[1]
n_classes = Y.shape[1]


# ||--- Neural Network Model ---||
def create_custom_model(input_dim, output_dim, _hparams, name='model'):
    # Create model function
    model = Sequential(name=name)
    for i in range(_hparams['n']):
        model.add(Dense(_hparams['nodes'], input_dim=input_dim, activation=_hparams['actv']))
        if _hparams['drop_enabled']:
            model.add(Dropout(rate=_hparams['drop_factor']))

    model.add(Dense(output_dim, activation='softmax'))

    # Compile model
    model.compile(loss=_hparams['loss'],  # fixed ->'categorical_crossentropy',
                  optimizer=_hparams['optimizer'],  # fixed -> 'adam',
                  metrics=['accuracy'])
    return model


def objective_function(model_params, batch_size, shuffle, epochs, verbose, validation_split, input_dim, output_dim):
    """Objective function for the Iris Classifier"""
    stats = {}

    # TensorBoard Callback
    cb = TensorBoard()
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=1e-5)

    model = create_custom_model(input_dim, output_dim, model_params)

    history_callback = model.fit(X_train, Y_train,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_split=validation_split,
                                 callbacks=[cb, early_stopping])
    score = model.evaluate(X_test, Y_test, verbose=0)

    stats['test_loss'] = score[0]
    stats['test_accuracy'] = score[1]

    Y_pred = model.predict(X_test)
    # from array of probabilities to array of 0-1
    # [0.872312,0.114322,0,013366] --> [1,0,0]
    for i, el in enumerate(Y_pred):
        i_max = int(np.argmax(Y_pred[i]))
        Y_pred[i].fill(0)
        Y_pred[i][i_max] = 1
    precision_W, recall_W, fscore_W, xyz = precision_recall_fscore_support(Y_test, Y_pred, average='weighted')
    stats['precision_W'] = precision_W
    stats['recall_W'] = recall_W
    stats['fscore_W'] = fscore_W

    score = score[0]  # we wnto to minimize the test loss
    return score, stats, {'model': model, 'history': history_callback}  # mandatory return format


# ||--- Domain Space ---||

# Domain Space
s = {
    'batch_size': hp.quniform('batch_size', 4, 16, 1),
    'shuffle': hp.choice('shuffle', [True, False]),
    'drop_enabled': hp.choice('drop_enabled',
                            [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 0.9, 0.1)},
                             {'drop_enabled': False, 'drop_factor': 0.1}]),
    'n': hp.quniform('n', 1, 12, 1),
    'nodes': hp.quniform('nodes', 4, 24, 2),
    'actv': 'relu',
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam'
    }

# Fixed Parameters
o_p = {'epochs': 200, 'validation_split': 0.15, 'verbose': 0, 'input_dim': 4, 'output_dim': 3}


def save_model_function(model_name, best_score, stats, model, history):
    model_name_ext = '{}/{}.h5'.format(trained_models_dir, model_name)
    model.save(model_name_ext)   # I want to save just the model(.h5 format) inside the 'trained_models' folder without take into consideration the history object
    return True  # it can return whatever we want


def s_f_p(params):
    drop_factor = params['drop_enabled'].get('drop_factor', 0.1)

    params['n'] = int(params['n'])
    params['nodes'] = int(params['nodes'])
    # Extract the drop_enabled
    params['drop_enabled'] = params['drop_enabled'][
        'drop_enabled']  # no second level dictionary, 'drop_factor' moved to the first level toghether with 'drop_enabled'
    params['drop_factor'] = round(drop_factor, 2)

    batch_size = int(params['batch_size'])  # int(params.pop('batch_size'))
    shuffle = params['shuffle']  # params.pop('shuffle')

    # the returned dictionary partially corresponds to the input of the objective function
    # what is missing is present in the dictionary of fixed parameters(o_p: epochs, verbose, validation_split, input_dim, output_dim)
    return {'model_params': params, 'batch_size': batch_size,
            'shuffle': shuffle}  # this output will be passed to the objective function


out_file = "iris_classifier_trials.csv"  # output file name of the csv file in which will be stored the <score, parameters, stats, iteration>
t_e = 200  # total_evals
s_t_e = 50  # save_trial_every
best, trial_fname = bayesian_optimization(function_to_optimize=objective_function, space_func_process=s_f_p,
                                          trial_fname=out_file, space=s, total_evals=t_e, save_trial_every=s_t_e,
                                          save_model_func=save_model_function, others_params=o_p)

print("\n\ntrial_fname: {}".format(trial_fname))
print("BEST parameter/s:")
print(best)