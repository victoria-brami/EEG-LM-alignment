import pandas as pd
import argparse
from config import Config as cfg
from kiloword.utils import read_table
from collections import Counter
from kiloword.utils import parse_table_labels

def create_semantic_labels_table():
    orig_df = read_table(cfg.LABELS_PATH)

    dest_df = parse_table_labels(orig_df, cfg.LIST_LABELS)
    semantic_df = {"#Words": [dest_df[dest_df[k] == True].count()[0] for k in cfg.LIST_LABELS]}
    semantic_csv = pd.DataFrame(data=semantic_df, index=cfg.LIST_LABELS)
    dest_df.to_csv(cfg.SEMANTICS_PATH, index=False)

def show_kiloword_dataset_stats():

    nb_labels_per_words = []
    nb_words_per_label = {}

    orig_data = read_table(cfg.LABELS_PATH)

    words = orig_data["WORD"]

    labels = read_table(cfg.SEMANTICS_PATH)
    for i, row in labels.iterrows():
        nb_labels_per_words.append(row.values.sum())
        if row.values.sum() == 0:
            print(words[i])

    for colname in labels.columns:
        nb_words_per_label[colname] = (labels[colname].values.sum())

    return nb_words_per_label, Counter(nb_labels_per_words)

if __name__ == '__main__':
    create_semantic_labels_table()
    w, r = show_kiloword_dataset_stats()
    print("Number of words per label: ", w)
    print("Number of labels per word: ", r)

