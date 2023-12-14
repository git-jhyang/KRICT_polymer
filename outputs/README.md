# Baseline models
files include model hyperparameters
- svr: Support vector regressor
- xgb: xgboost
- fp: rdkit fingerprints
- gg: graph & global state feature

# Neural network models (finetune / scratch)
Models that exhibit highest R2 scores with given condition are collected
- all_ens  : regardless of hyperparameters and input features
- gr_ens   : regardless of hyperparameters but only use graph features
- gr_hp    : same hyperparameters and use of graph features
- grgs_ens : regardless of hyperparameters but use both graph and global state features
- grgs_hp  : same hyperparameters and use of both graph and global state features
- gs_ens   : regardless of hyperparameters but only use global state features
- gs_hp    : same hyperparameters and use of global state features

- prefixes:
-- gr: graph
-- gs: global state
-- grgs: graph and global state
- suffixes:
-- ens: ensemble regardless of hyperparameter, individual R2 score
-- hp: same hyperparameter, average of R2 score

## file description
tuning_hp.txt : abstract hyperparameter information

prefix: 
    fold_N : results from N-th fold
suffixes:
    loss.txt    : train and validation loss
    model.torch : pytorch model
    param.json  : model hyperparameter
    params.txt  : final learning rate and score
    scale.json  : target scaling factors
