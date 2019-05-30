import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

call_name_svded_features = pd.read_pickle("call_name_svded_features.pkl")
api_name_svded_features = pd.read_pickle("api_name_svded_features.pkl")
exinfos_svded_features = pd.read_pickle("exinfos_svded_features.pkl")
merge_data = np.hstack([api_name_svded_features, exinfos_svded_features, call_name_svded_features])

for cluster in [50, 250, 300, 400, 500]:
    kmeans = KMeans(n_clusters=cluster, random_state=0)
    y_pred = kmeans.fit_predict(merge_data)
    result = pd.DataFrame()
    result["id"] = pd.read_csv("id.csv", names=["id"])["id"]
    result["family_id"] = y_pred

    result.to_csv(f"k-means_cluster={cluster}_result.csv", encoding="utf-8", index=False)