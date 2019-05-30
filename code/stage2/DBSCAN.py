import numpy as np
import pandas as pd
import sklearn.cluster as skc

api_name_svded_train = pd.read_pickle("api_name_svded_features.pkl")
exinfos_svded_train = pd.read_pickle("exinfos_svded_features.pkl")
call_name_svded_train = pd.read_pickle("call_name_svded_features.pkl")

merge_data = np.hstack([api_name_svded_train, exinfos_svded_train, call_name_svded_train])
dbscan = skc.DBSCAN()
y_pred = dbscan.fit_predict(merge_data)

result = pd.DataFrame()
result["id"] = pd.read_csv("id.csv", names=["id"])["id"]
result["family_id"] = y_pred

result.to_csv("result.csv", encoding="utf-8", index=False)
