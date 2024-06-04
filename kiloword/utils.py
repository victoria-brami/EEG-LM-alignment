import pandas as pd
import numpy as np
import mne
# import spacy
from .config import Config


def read_table(table_path: str):
    return pd.read_csv(table_path)

def save_table(df: pd.DataFrame, dest_path: str):
    return df.to_csv(dest_path, index=False)

def load_kiloword_metadata(datapath: str = Config().METADATA) -> pd.DataFrame:
    """

    :param datapath: Path to the dataset's metadata (a txt file)
    :return: table containing all metadata info
    """
    metadata = open(datapath, "r")
    columns = metadata.readline().strip().split()
    metadata_df = pd.DataFrame(columns=columns)
    for line in metadata.readlines():
        data = line.strip().split()
        metadata_df.loc[len(metadata_df)] = data
    return metadata_df

def load_data_from_fif(datapath: str=Config().DATA):
    raw = mne.read_epochs(datapath)
    list_features = []

    data = raw.copy().get_data()

    for i in range(data.shape[0]):
        list_features.append(data[i].reshape(-1))

    return np.array(list_features)


def normalize_data(data: np.array, mode: str="min_max"):
    if mode == "min_max":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    elif mode == "normal":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
    return scaler.fit_transform(data)


# def tag_words_pos(list_words: list):
#     nlp = spacy.load("en_core_web_sm")
#     list_pos = []
#     for word in list_words:
#         doc = nlp(word)
#         for token in doc:
#             list_pos.append(token.pos_)
#     return list_pos
