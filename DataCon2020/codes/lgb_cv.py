import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


# LightGBM模型+交叉验证
params =  {'boosting_type': 'gbdt',
           'objective': 'binary',
           'metric': 'binary_logloss',
           'learning_rate': 0.001,
           'num_leaves': 82,
           'max_depth': 8,
           'min_data_in_leaf': 64,
           'min_child_weight':1.435,
           'bagging_fraction': 0.785,
           'feature_fraction': 0.373,
           'bagging_freq': 22,
           'reg_lambda': 0.065,
           'reg_alpha': 0.797,
           'min_split_gain': 0.350,
           'nthread': 8,
           'seed': 42,
           'scale_pos_weight':1.15,
           'verbose': -1}

def get_lgb_oof(params, x_train, y_train, x_test, n_splits):
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    kf = StratifiedKFold(n_splits=n_splits, random_state=0)
    oof_train = np.empty((n_train, ))
    oof_test = np.empty((n_test, ))
    oof_test_skf = np.empty((n_splits, n_test))
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_test = x_train[test_index]
        kf_y_test = y_train[test_index]
        train_matrix = lgb.Dataset(kf_x_train, label=kf_y_train)
        valid_matrix = lgb.Dataset(kf_x_test, label=kf_y_test)
        model = lgb.train(params, 
                          train_set=train_matrix, 
                          num_boost_round=80000, 
                          valid_sets=valid_matrix, 
                          verbose_eval=-1, 
                          early_stopping_rounds=600)
        oof_test_skf[i, :] = model.predict(x_test)
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_test.reshape(-1, 1)
