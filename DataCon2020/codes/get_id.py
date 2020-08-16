import glob
import pandas as pd

names = []
df = pd.DataFrame()
for path in glob.glob("/home/datacon/malware/YYY_step1/*"):
    names.append(path.split("/")[-1])

df["id"] = names
df.to_csv("/home/jovyan/media_directory/test_id.csv", index=False, header=None, encoding="utf-8")