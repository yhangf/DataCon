import re
import glob
import pandas as pd

def get_string(directory, file_name):
    list_ = []
    df = pd.DataFrame()
    for path in glob.glob(f"{directory}/*"):
        with open(path, "rb") as fp:
            string = fp.read().decode("utf-8", errors="ignore")
        raw_words = re.findall("[a-zA-Z]+", string) 
        words_space = " ".join(w for w in raw_words if 4 < len(w) < 20)
        list_.append(words_space)
    df["words"] = list_
    df.to_csv(f"{file_name}.csv", index=False)
    print(len(list_))


get_string("/home/datacon/malware/YYY_step1", "/home/jovyan/media_directory/end_raw_test")