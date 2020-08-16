import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


with open("/home/jovyan/models/tfidf_model", "rb") as fp:
    vectorizer = joblib.load(fp)
with open("/home/jovyan/models/train_model", "rb") as fp:
    model = joblib.load(fp)

test_data_ = pd.read_csv("/home/jovyan/media_directory/end_raw_test.csv")
id_ = pd.read_csv("/home/jovyan/media_directory/test_id.csv", header=None)

test_tfidf_features = vectorizer.transform(test_data_.words.tolist())
y_pred = model.predict(test_tfidf_features)

result = pd.DataFrame()
result["id_"] = id_.values.flatten()
result["y_pred"] = y_pred

result.to_csv("/home/jovyan/malware_final.txt", index=False, header=None)