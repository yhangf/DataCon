import pickle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD 

call_name_svded_features = pd.read_pickle("call_name_svded_features.pkl")
api_name_svded_features = pd.read_pickle("api_name_svded_features.pkl")
exinfos_svded_features = pd.read_pickle("exinfos_svded_features.pkl")
merge_data = np.hstack([api_name_svded_features, exinfos_svded_features, call_name_svded_features])
X_tsne = TSNE(n_components=2, random_state=33).fit_transform(merge_data)

with open("api_name_exinfos_tsne_call_name_data.pkl", "wb") as fp:
    pickle.dump(X_tsne, fp)