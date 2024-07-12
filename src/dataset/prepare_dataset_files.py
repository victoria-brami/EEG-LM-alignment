import os
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from collections import Counter


def get_sentences_from_session(datapath: str, session_folder: str) -> None:
    """

    :param datapath: Path to the folder containing the data
    :param session_folder: folder containing the recordings related to a single session (6 in total for UBIRA)
    :return: gets the sentences read during the session (only on word-level in input) and stores them in a csv file
            in the same session_folder
    """

    def get_sort_key(path):
        return int(path.split('_files')[1].split('-')[0])

    src_path = os.path.join(datapath, session_folder)
    list_fif_files = sorted(glob(os.path.join(src_path, '*.fif')), key=get_sort_key)

    dest_df = {"session_id": [], "sent_ident": [], "sent_content": [], "sent_num": [], "fif_file": []}
    for fif_file in tqdm(list_fif_files):
        epochs = mne.read_epochs(f"{fif_file}", preload=True, verbose=50)
        grouped_labels = epochs.metadata.groupby("sent_ident")
        for sent_ident, group in grouped_labels:
            dest_df["sent_ident"].append(sent_ident)
            dest_df["fif_file"].append(os.path.basename(fif_file))
            dest_df["session_id"].append(session_folder)
            dest_df["sent_content"].append(" ".join(group["word"].to_list()))
            dest_df["sent_num"].append(str(pd.unique(group["sent_num"])[0]))
    dest_df = pd.DataFrame(dest_df)
    dest_df.to_csv(f"{src_path}/all_sentences.csv", index=False)


def get_unique_and_single_sentences_from_session(datapath: str, session_folder: str) -> None:
    """

    :param datapath: Path to the folder containing the data
    :param session_folder: folder containing the recordings related to a single session (6 in total for UBIRA)
    :return:
    """
    # Load the csv file containing all the sentences (computed with previous function)
    src_path = os.path.join(datapath, session_folder)
    sentence_df = pd.read_csv(os.path.join(src_path, "all_sentences.csv"))
    list_sentences = sentence_df["sent_content"].values

    counted_list_sentences = dict(Counter(list_sentences))

    # Separate Unique and duplicate sentences and save them in different files
    unique_list = []
    duplicate_list = []

    for sentence, count in counted_list_sentences.items():
        if count >= 2:
            duplicate_list.append(sentence)
        else:
            unique_list.append(sentence)

    unique_sent_df = pd.DataFrame(columns=sentence_df.columns)
    duplicate_sent_df = pd.DataFrame(columns=sentence_df.columns)

    for i, row in sentence_df.iterrows():
        if row["sent_content"] in duplicate_list:
            duplicate_sent_df.loc[len(duplicate_sent_df)] = row
        else:
            unique_sent_df.loc[len(unique_sent_df)] = row

    unique_sent_df.to_csv(os.path.join(src_path, "all_unique_sentences.csv"), index=False)
    duplicate_sent_df.to_csv(os.path.join(src_path, "all_duplicate_sentences.csv"), index=False)


def label_all_sentences_from_sessions(datapath: str):
    list_folders = [os.path.join(datapath, fold) for fold in os.listdir(datapath)
                    if os.path.isdir(os.path.join(datapath, fold))]
    lfold = sorted([fold for fold in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, fold))])
    fif_fold = [f"{fold}_fif_name" for fold in lfold]

    list_sent_df = [pd.read_csv(os.path.join(fold,  "all_unique_sentences.csv")) for fold in list_folders]
    sent_df = pd.concat(list_sent_df)
    sent_occurences = dict(Counter(sent_df["sent_content"]))
    sent_occurences = dict(sorted(sent_occurences.items(), key=lambda item: item[1], reverse=True))

    sentences_indices = pd.DataFrame(columns=["common_id", "sent_content", *lfold, *fif_fold])

    for i, sent in enumerate(sent_occurences.keys()):
        # Initialize the row to append to the csv
        row = [f"common_sent_{i}", sent, *[""] * len(lfold), *[""] * len(fif_fold)]
        # Get the sessions in which the sentence appears
        sent_rows = sent_df[sent_df["sent_content"] == sent]
        for session in sent_rows["session_id"].values:
            sess_id = int(session.split("_")[-1])
            row[sess_id + 2] = sent_rows[sent_rows["session_id"] == session]["sent_ident"].values[0]
            row[sess_id + 2 + len(lfold)] = sent_rows[sent_rows["session_id"] == session]["fif_file"].values[0]
        sentences_indices.loc[len(sentences_indices)] = row

    sentences_indices.to_csv(os.path.join(datapath, "sentences_indices.csv"), index=False)


def create_sentence_level_files(datapath: str):

    def retrieve_session_id_from_name(filename: str):
        return filename.split("_")[1]

    # def retrieve_fif_file_from_sent_id(sent_name_id: str):
        # sess_0_newsgroup-groups.google.com_hiddennook_f50294175d32a8ac_ENG_20041120_152800.txt_sent_40

    all_sentence_labels = pd.read_csv(os.path.join(datapath, "sentences_indices.csv"))
    #
    # list_folders = [os.path.join(datapath, fold) for fold in os.listdir(datapath)
    #                 if os.path.isdir(os.path.join(datapath, fold))]

    session_cols = sorted([fold for fold in all_sentence_labels.columns
                           if fold.startswith("session") and not fold.endswith("_fif_name")])
    session_fif_cols = sorted([fold for fold in all_sentence_labels.columns
                               if fold.startswith("session") and  fold.endswith("_fif_name")])

    for i, row in tqdm(all_sentence_labels.iterrows()):
        sent_common_id = row["common_id"]
        if i < 109:
            continue
        print("sentence ", i)
        for sess_name, fif_id in zip(session_cols, session_fif_cols):
            if sess_name != "" or len(sess_name) > 0:
                session_folder = f"session_{retrieve_session_id_from_name(sess_name)}"

                # Create the sentence folder if it does not exist
                dest_folder = os.path.join(datapath, session_folder, sent_common_id)
                os.makedirs(dest_folder, exist_ok=True)

                # Retrieve the data in the right fif file and extract the sentence-wise data
                fif_name = os.path.join(datapath, session_folder, row[fif_id])
                epochs = mne.read_epochs(fif_name, preload=True, verbose=50)
                eeg_data = epochs.get_data()

                epochs.metadata.index = range(len(epochs.metadata))
                labels_df = epochs.metadata[epochs.metadata["sent_ident"] == row[sess_name]]
                word_in_sent_ids = labels_df.index.to_numpy()
                
                sent_eeg = eeg_data[word_in_sent_ids]
                np.save(os.path.join(dest_folder, f"{sent_common_id}_eeg.npy"), sent_eeg)
                labels_df.to_csv(os.path.join(dest_folder, f"{sent_common_id}_labels.csv"), index=False)

def process_data(datapath: str, session_folder: str, filename: str) -> None:
    """

    :param datapath:
    :param session_folder:
    :param filename:
    :return:
    """

    all_src_path = os.path.join(datapath, session_folder, filename)
    epochs = mne.read_epochs(all_src_path, verbose=50)
    epochs.get_data()
    eeg_data_df = epochs.metadata

    eeg_dest_path = all_src_path.replace(".fif", ".npy")
    eeg_labels_dest_path = all_src_path.replace(".fif", ".csv")

    # np.save(eeg_dest_path, eeg_data)
    # eeg_data_df.to_csv(eeg_labels_dest_path, index=False)


def load_processed_data(datapath: str, session_folder: str, filename: str) -> np.array:
    src_path = os.path.join(datapath, session_folder, filename)

    return np.load(src_path)


if __name__ == '__main__':
    rootpath = "/home/viki/Documents/Erasmus/data_playground/eeg_POS"
    list_folders = ["session_0", "session_1", "session_2", "session_3", "session_4", "session_5"]
    session_filename = "rsvp_session1_files10-18-epo.fif"
    # process_data(datapath=datapath, session_folder=folder, filename=filename)

    # for sess_folder in tqdm(list_folders[2:], desc=f"Get all sentences from each session..."):
    #     # Create csv file containing all the sentences read within a single session
    #     get_sentences_from_session(rootpath, sess_folder)
    #     # Filter and keep sentences appearing only ONCE within a single session
    #     get_unique_and_single_sentences_from_session(rootpath, sess_folder)

    # # Get the Unique sentences and Create common sentence id across the different sessions
    # # (will be easier later if we want to get the averaged eeg signals over the sessions )
    # label_all_sentences_from_sessions(rootpath)

    # Create the labels and signal files for each of those sessions (compute for all the sentences at first,
    # then isolate the duplicate sentences )
    create_sentence_level_files(rootpath)


