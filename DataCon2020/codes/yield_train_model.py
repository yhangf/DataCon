import pandas as pd
from xgboost import XGBClassifier
import joblib



train_tfidf_features = pd.read_pickle("/home/jovyan/media_directory/train_tfidf_features")
labels = pd.read_pickle("/home/jovyan/media_directory/train_labels")

model = XGBClassifier(n_estimators=400, learning_rate=0.05) 
model.fit(train_tfidf_features, labels)

with open("/home/jovyan/models/train_model", "wb") as fp:
    joblib.dump(model, fp)