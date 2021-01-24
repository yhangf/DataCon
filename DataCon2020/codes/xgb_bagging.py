import numpy as np
from xgboost import XGBClassifier

# 适用于训练数据较少，且预测值抖动现象明显的场合
result = []
for i in np.random.randint(0xFFFFF, size=10):
    train_data, test_data, train_label, test_label = train_test_split(train_tfidf_features, 
                                                                      labels, 
                                                                      test_size=0.2, 
                                                                      random_state=i)

    model = XGBClassifier(n_estimators=100) 
    model.fit(train_data, train_label)
    y_pred = model.predict(test_tfidf_features)
    result.append(y_pred)
y_pred = np.array(result).mean(axis=0)
y_pred_end = [1 if i >= 0.5 else 0 for i in y_pred]
