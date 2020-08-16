import joblib
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

train_data_ = pd.read_csv("media_directory/raw_train_data.csv")

vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, max_features=3000)
train_tfidf_features = vectorizer.fit_transform(train_data_.words.tolist())

with open("/home/jovyan/models/tfidf_model", "wb") as fp:
    joblib.dump(vectorizer, fp)
    
with open("/home/jovyan/media_directory/train_tfidf_features", "wb") as fp:
    pickle.dump(train_tfidf_features, fp)
    
with open("/home/jovyan/media_directory/train_labels", "wb") as fp:
    pickle.dump(train_data_.labels, fp)