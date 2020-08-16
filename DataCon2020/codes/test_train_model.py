import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def calc_score(y_true, y_pred, alpha=1.2):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_black_index = {i for i in range(len(y_true)) if y_true[i] == 1}
    y_pred_black_index = {i for i in range(len(y_pred)) if y_pred[i] == 1}
    y_true_white_index = {i for i in range(len(y_true)) if y_true[i] == 0}
    y_pred_white_index = {i for i in range(len(y_pred)) if y_pred[i] == 0}
    
    black_is_black = len(y_true_black_index & y_pred_black_index)
    black_is_white = len(y_true_black_index & y_pred_white_index)
    white_is_black = len(y_true_white_index & y_pred_black_index)
    white_is_white = len(y_true_white_index & y_pred_white_index)
    
    recall = black_is_black / (black_is_black + black_is_white) 
    error_ratio = white_is_black / (white_is_black + white_is_white)
    score = recall - alpha * error_ratio
    return score

train_data_ = pd.read_pickle("/home/jovyan/media_directory/train_tfidf_features")
train_labels = pd.read_pickle("/home/jovyan/media_directory/train_labels")

result = []

for i in range(20, 50):
    train_data, test_data, train_label, test_label = train_test_split(train_data_, 
                                                                      train_labels, 
                                                                      test_size=0.25, 
                                                                      random_state=i)

    _ = []
    model = XGBClassifier(max_depth=5, n_estimators=90) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    score = calc_score(test_label, y_pred)
    _.append(score)

    model = XGBClassifier(max_depth=5, n_estimators=80) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)

    score = calc_score(test_label, y_pred)
    _.append(score)
    
    model = XGBClassifier(max_depth=5, n_estimators=70) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    score = calc_score(test_label, y_pred)
    _.append(score)

    model = XGBClassifier(max_depth=5, n_estimators=60) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    score = calc_score(test_label, y_pred)
    _.append(score)

    model = XGBClassifier(max_depth=5, n_estimators=50) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    score = calc_score(test_label, y_pred)
    _.append(score)

    model = XGBClassifier(max_depth=5, n_estimators=40) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    score = calc_score(test_label, y_pred)
    _.append(score)

    model = XGBClassifier(max_depth=5, n_estimators=30) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_data)
    score = calc_score(test_label, y_pred)
    _.append(score)
    
    result.append(_)
    
print(np.vstack(result).mean(axis=0))

