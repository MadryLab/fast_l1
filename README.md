# Fast GPU-Enabled LASSO

A fast SAGA-based GPU solver for L1-regularized regression. 

Usage:
```python

from fast_l1 import regressor
# loader yields (inp, targ, index), all on GPU
train_loader = ...
val_loader = ...

max_lam = regressor.calc_max_lambda(train_loader)

weight = ch.zeros(n_features, n_targets).cuda()
bias = ch.zeros(n_targets).cuda()

eps = 1e-6 # Factor btw max and min lambda
k = 100 # Number of lambdas to try
kwargs = {
    # Learning rate
    'lr': 0.01,
    # Starting lambdas
    'start_lams': max_lam,
    # How much to decay lambda by upon convergence
    'lam_decay': np.exp(np.log(eps)/k),
    # Number of lambdas to try
    'num_lambdas': k,
    # How often to evaluate on test set
    'early_stop_freq': 5,
    # Threshold for optimizer convergence
    'early_stop_eps': 5e-10,
    # Logging directory
    'logdir': LOG_DIR
}
regressor.train_saga(weight,
                     bias,
                     train_loader,
                     val_loader,
                     **kwargs)
```

Any questions? Open an issue or email datamodels@mit.edu.