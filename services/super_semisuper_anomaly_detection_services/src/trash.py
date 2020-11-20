import matplotlib as plt
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
import seaborn as sns
import numpy as np


# Create the learning rate
learning_rate = {'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.2))}
learning_rate_dist = []

# Draw 10000 samples from the learning rate domain
for _ in range(10000):
    learning_rate_dist.append(sample(learning_rate)['learning_rate'])

plt.figure(figsize=(8, 6))
sns.kdeplot(learning_rate_dist, color='red', linewidth=2, shade=True)
plt.title('Learning Rate Distribution', size=18)
plt.xlabel('Learning Rate', size=16)
plt.ylabel('Density', size=16)
plt.show()

# Discrete uniform distribution
batch_size = {'batch_size': hp.quniform('batch_size', 8, 64, 8)}
batch_size_dist = []

# Sample 10000 times from the number of leaves distribution
for _ in range(10000):
    batch_size_dist.append(sample(batch_size)['batch_size'])

# kdeplot
plt.figure(figsize=(8, 6))
sns.kdeplot(batch_size_dist, linewidth=2, shade=True)
plt.title('Batch Size Distribution', size=18)
plt.xlabel('Batch Size', size=16)
plt.ylabel('Density', size=16)
plt.show()


# drop_enabled domain
drop_enabled = {'drop_enabled': hp.choice('drop_enabled',
                                         [{'drop_enabled': True, 'drop_factor': hp.quniform('drop_factor', 0.1, 1, 0.1)},
                                         {'drop_enabled': False, 'drop_factor': 0.1}])}

params = sample(drop_enabled)
# Retrieve the subsample if present otherwise set to 0.1
drop_factor = params['drop_enabled'].get('drop_factor', 0.1)

# Extract the boosting type
params['drop_enabled'] = params['drop_enabled']['drop_enabled']
params['drop_factor'] = drop_factor
print(params)