# Fast GPU-Enabled LASSO

A fast SAGA-based GPU solver for L1-regularized regression. 
Usage:
```python

from fast_l1 import regressor
# loader must return (input, target, index), all on GPU
train_loader = ...
val_loader = ...

max_lam = regressor.calc_max_lambda(train_loader)

weight = ch.zeros(n_features, n_targets).cuda()
bias = ch.zeros(n_targets).cuda()

regressor.train_saga(weight,
                     bias,
                     train_loader,
                     val_loader,
                     lr=lr,
                     start_lams=max_lam,
                     lam_decay=np.exp(np.log(eps)/k),
                     num_lambdas=k,
                     early_stop_freq=early_stop_freq,
                     early_stop_eps=early_stop_eps,
                     logdir=LOG_DIR)
```