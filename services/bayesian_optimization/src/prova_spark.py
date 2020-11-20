from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from hyperopt import fmin, hp, tpe
from hyperopt import SparkTrials, STATUS_OK

# Load MNIST data, and preprocess it by standarizing features.
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=5000, test_size=10000)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# First, set up the scikit-learn workflow, wrapped within a function.
def train(params):
  """
  This is our main training function which we pass to Hyperopt.
  It takes in hyperparameter settings, fits a model based on those settings,
  evaluates the model, and returns the loss.

  :param params: map specifying the hyperparameter settings to test
  :return: loss for the fitted model
  """
  print("CHIAMATA!")
  # We will tune 2 hyperparameters:
  #  regularization and the penalty type (L1 vs L2).
  regParam = float(params['regParam'])
  penalty = params['penalty']

  # Turn up tolerance for faster convergence
  clf = LogisticRegression(C=1.0 / regParam,
                           multi_class='multinomial',
                           penalty=penalty, solver='saga', tol=0.1)
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)

  return {'loss': -score, 'status': STATUS_OK}

# Next, define a search space for Hyperopt.
search_space = {
  'penalty': hp.choice('penalty', ['l1', 'l2']),
  'regParam': hp.loguniform('regParam', -10.0, 0),
}

# Select a search algorithm for Hyperopt to use.
algo=tpe.suggest  # Tree of Parzen Estimators, a Bayesian method

'''
# We can run Hyperopt locally (only on the driver machine)
# by calling `fmin` without an explicit `trials` argument.
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=32)
best_hyperparameters
'''
# We can distribute tuning across our Spark cluster
# by calling `fmin` with a `SparkTrials` instance.
spark_trials = SparkTrials(parallelism=8)
best_hyperparameters = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  trials=spark_trials,
  max_evals=32)
best_hyperparameters