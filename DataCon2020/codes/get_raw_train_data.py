import re
import pandas as pd

def get_train_data(label_file_path, data_db_path, file_name, media_directory):
    with open(label_file_path, "r") as fp:
        id_ = fp.read().split()
    list_ = []
    df = pd.DataFrame()
    for path in id_:
        with open(f"{data_db_path}/{path}", "rb") as fp:
            string = fp.read().decode("utf-8", errors="ignore")
        raw_words = re.findall("[a-zA-Z]+", string) 
        words_space = " ".join(w for w in raw_words if 4 < len(w) < 20)
        list_.append(words_space)
    df["words"] = list_
    df.to_csv(f"{media_directory}/{file_name}.csv", index=False)
    return df

def merge(black, white, file_name, media_directory):
    train_raw_data = black.append(white)
    train_raw_data["labels"] = [1 for _ in range(black.shape[0])] + [0 for _ in range(white.shape[0])]
    train_raw_data.to_csv(f"{media_directory}/{file_name}.csv", index=False)
    
black = get_train_data("/home/datacon/malware/XXX/black.txt",
                       "/home/datacon/malware/XXX/data",
                       "black",
                       "/home/jovyan/media_directory")
print("black is over!")
white = get_train_data("/home/datacon/malware/XXX/white.txt",
                       "/home/datacon/malware/XXX/data",
                       "white",
                       "/home/jovyan/media_directory")
print("white is over!")
# black = pd.read_csv("/home/jovyan/media_directory/black.csv")
# white = pd.read_csv("/home/jovyan/media_directory/white.csv")
merge(black, white, "raw_train_data", "/home/jovyan/media_directory")