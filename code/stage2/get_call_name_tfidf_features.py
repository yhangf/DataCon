import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv("call_name.csv")
call_name_vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9)
call_name_train_tfidf_features = call_name_vectorizer.fit_transform(data["call_name"].tolist())
with open("call_name_tfidf_features.pkl", "wb") as fp:
    pickle.dump(call_name_train_tfidf_features, fp)
    
svd = TruncatedSVD(n_components=1000, algorithm="arpack", random_state=0)
call_name_svded_train = svd.fit_transform(call_name_train_tfidf_features.tolil())

with open("call_name_svded_features.pkl", "wb") as fp:
    pickle.dump(call_name_svded_train, fp)
    
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(call_name_svded_train)

with open("call_name_tsne_data.pkl", "wb") as fp:
    pickle.dump(X_tsne, fp)