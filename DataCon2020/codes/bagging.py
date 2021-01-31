import numpy as np
from sklearn.model_selection import StratifiedKFold

def bagging(model, x_train, y_train, x_test, n_splits):
    """
    :@param x_train: feature matrix.
    :type x_train: np.array(M X N) or list(M X N).
    :@param y_train: class label.
    :type y_train: np.array(M X 1).
    :@param x_test: test set feature matrix.
    :type x_test: np.array(M X N) or list(M X N).
    :@param n_splits: K-fold parameter.
    :type n_splits: int.
    """
    n_train, n_test = x_train.shape[0], x_test.shape[0]
    # 随机划分数据
    kf = StratifiedKFold(n_splits=n_splits, random_state=0)
    oof_train = np.empty((n_train, ))
    oof_test = np.empty((n_test, ))
    oof_test_skf = np.empty((n_splits, n_test))

    # 训练第i个模型
    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        model.fit(kf_x_train, kf_y_train)
        oof_test_skf[i, :] = model.predict(x_test)
    # 对所有的模型结果进行集成
    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_test.reshape(-1, 1)
