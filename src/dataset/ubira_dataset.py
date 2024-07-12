import os
from glob import glob
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.signal import hilbert
from omegaconf import OmegaConf
from torch.utils.data import Dataset


def _split_sent_data_into_words_data(start_id: int,
                                     sent_id: str,
                                     eeg_sig: np.array,
                                     labels_df: pd.DataFrame,
                                     filter_pos: Optional[List[str]] = None):

    word_data_list = []
    sentence = " ".join(labels_df["word"].values)
    for i, row in labels_df.iterrows():
        sample = {
            "id": start_id + i,  # Unique key to define the sample
            "sent_id": sent_id,
            "word": row["word"],
            "sentence": sentence,
            "raw_eeg_input_ids": eeg_sig[i],
            "pos": row["pos"],
            "prev_pos": row["prev_pos"],
            "next_pos": row["next_pos"],
            "freq": row["freq"],
            "prev_freq": row["prev_freq"],
            "next_freq": row["next_freq"],
            "len": row["len"],
            "prev_len": row["prev_len"],
            "next_len": row["next_len"],
        }
        # Apply Specific label filtering if needed
        if filter_pos is not None and sample["pos"] not in filter_pos:
            continue
        word_data_list.append(sample)

    return word_data_list


def _compute_eeg_power_signal(sent_eeg: np.array):
    """

    :param sent_eeg: "2D array of shape (N_WORDS X TIMESTEPS, Channels)
    :return:
    """
    return np.abs(hilbert(sent_eeg) * np.conjugate(hilbert(sent_eeg)))


class UBIRADataset(Dataset):

    def __init__(self, config, tokenizer=None):
        self.config = config
        self.datapath = config.root_path
        self.folders = config.list_folders
        self.tokenizer = tokenizer
        self.eegs = []
        self.meta_data = []
        self._load_sentence_labels()
        self._load_channels()
        self._load_data(mode=config.mode, use_power=config.use_power)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _load_sentence_labels(self):
        self.sentence_labels = pd.read_csv(os.path.join(self.datapath, "sentences_indices.csv"))

    def _load_channels(self):
        self.channels = pd.read_csv(os.path.join(self.datapath, "locs3d.csv"))

    def _get_sentence_wise_data(self, sent_id: str, mode: str = "average", power: bool = True) -> List:
        list_paths = glob(os.path.join(self.datapath, "*", f"{sent_id}", "*.npy"))
        labels_path = list_paths[0].replace("eeg.npy", "labels.csv")

        labels_df = pd.read_csv(labels_path)
        labels_df = labels_df[["word", "pos", "filename", "freq", "len",
                               "prev_pos", "next_pos", "prev_freq", "next_freq", "prev_len", "next_len"]]

        # We average all the power
        if mode == "average":
            # Array of shape (n_sessions, n_words, n_channels, n_timesteps)
            eeg_sigs_arr = np.stack([np.load(path) for path in list_paths])
            n_words, channels, timesteps = eeg_sigs_arr[0].shape

            if not power:
                eeg_sig = np.mean(eeg_sigs_arr, axis=0)

            else:
                # we want to compute the power so we reshape the signal
                eeg_sigs_arr = eeg_sigs_arr.transpose(0, 2, 1, 3)
                # Reshape the array to (n_sessions,  n_channels, n_words * n_timesteps)
                eeg_sigs_arr = eeg_sigs_arr.reshape(-1, channels, n_words * timesteps)
                # Apply the Power computation for each trial
                eeg_pows_arr = _compute_eeg_power_signal(eeg_sigs_arr)
                eeg_pows_arr = eeg_pows_arr.reshape(-1, channels, n_words, timesteps).transpose(0, 2, 1, 3)
                eeg_sig = np.mean(eeg_pows_arr, axis=0)
            return sent_id, eeg_sig, labels_df

    def _load_data(self, mode: str = "word", use_power: bool = True):
        """

        :param mode: (str) whether we load the signals on the sentence of word level
        :return: the data from the specified folder (6 in this dataset)
        """
        self.data = []
        self.start_id = 0

        # Iterate on sentence level
        for sent_i, row in self.sentence_labels.iterrows():
            if sent_i > self.config.n_sentences:
                break
            sent_id_name = row["common_id"]

            _, eeg_sig, labels_df = self._get_sentence_wise_data(sent_id_name, mode="average", power=use_power)

            if mode == "word":
                word_data_list = _split_sent_data_into_words_data(self.start_id + sent_i,
                                                                  sent_id_name,
                                                                  eeg_sig,
                                                                  labels_df,
                                                                  filter_pos=self.config.labels)
                self.data.extend(word_data_list)
                self.start_id += len(word_data_list) - 1

            elif mode == "sentence":
                continue


if __name__ == '__main__':
    conf = OmegaConf.load("../../tests/config.yaml")
    config =  conf.dataset
    dataset = UBIRADataset(config)
    print("Data Length", len(dataset), "\n")
    print([dataset[i]["id"] for i in range(len(dataset))])
    print(dataset[86])