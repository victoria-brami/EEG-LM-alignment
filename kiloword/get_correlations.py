import argparse
import os.path

import numpy as np
import pandas as pd

from kiloword.config import Config as cfg
from kiloword.utils import read_table, parse_table_labels
from kiloword.analysis import (
    all_pairs,
    get_model_representations,
    compute_all_representations_distances,
    compute_all_dl_distance,
    compute_correlations
)
from kiloword.evaluation import CorrelationsTable

LIST_LABELS = ["ENTERTAINMENT", "MONEY", "NATURE", "QUANTITY",
               "POLITICS", "RELIGION", "HOUSE", "MOVE", "SPORT",
               "JUSTICE", "INDUSTRY", "LANGUAGE", "FOOD", "MODE",
               "DEVICE", "FAMILY", "MUSIC", "CRIME", "CATASTROPHE",
               "ARMY", "TIME", "SCHOOL", "CLEANNESS", "DEATH",
               "GLORY", "BODY", "PEOPLE", "MEDICAL", "MATERIAL",
               "GOVERN", "SCIENCE", "PHILOSOPHY"]


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_label", type=str,
                        default=None,
                        # choices=LIST_LABELS,
                        help="compute correlations of a specific semantic field")
    parser.add_argument("--save_folder", type=str,
                        default="/home/viki/Downloads/kiloword_correlations",
                        help="folder where the experiments are saved")
    parser.add_argument("--tab_name", type=str,
                        default="correlations.csv",
                        help="Name of the document where the experiments are saved")
    parser.add_argument("--labels_path", type=str,
                        default=cfg.LABELS_PATH,
                        help="File containing the annotations")
    parser.add_argument("--use_model_cache", type=bool, action="store_true",
                        help="whether to load pre-computed word representations or not")
    parser.add_argument("--eeg_path", type=str,
                        default=cfg.DATA,
                        help="File containing the EEG recordings")
    parser.add_argument("--word_dist_repr", type=str,
                        default="bert",
                        choices=["bert", "bert_random", "hubert", "hubert_random",
                                 "levenshtein", "levenshtein_ipa"],
                        help="Word representations type")
    parser.add_argument("--tab_attrs", type=list,
                        default=['Channel', 'distance', 'truncate_start',
                                 'truncate_end', 'pearson', 'spearman'],
                        nargs="+",
                        help="keys to figure in the saved documents")
    parser.add_argument("--pad_step", type=int,
                        default=10,
                        help="padding step")
    parser.add_argument("--timesteps", type=int,
                        default=31,
                        help="duration of the eeg signals extracted")
    return parser.parse_args()


def main(args):
    # Download the labels
    labels = read_table(args.labels_path)
    labels_table = parse_table_labels(labels, LIST_LABELS, labelcolname="SEMANTIC_FIELD")

    all_ids = np.arange(len(labels))

    if args.focus_label is not None:
        if args.focus_label == "OBJECT":
            all_ids = labels[labels["MATERIAL"] == "YES"].index
        elif args.focus_label == "ABSTRACT":
            all_ids = labels[labels["MATERIAL"] != "YES"].index
        else:
            all_ids = labels_table[labels_table[args.focus_label] == True].index
        args.tab_name = "_".join([args.focus_label, args.tab_name])
    else:
        args.tab_name = "_".join(["ALL", args.tab_name])

    args.tab_name = "_".join([args.word_dist_repr, args.tab_name])
    # Get the list of words and their pairs
    list_words = labels["WORD"].values[all_ids]
    list_paired_words = all_pairs(list_words)
    list_paired_indices = all_pairs(range(len(list_words)))

    # Initialize Experiment table
    corr = CorrelationsTable(name=args.tab_name,
                             table_folder=args.save_folder,
                             table_columns=args.tab_attrs)

    # Download the EEG data and drop non-useful info
    eeg_data = read_table(args.eeg_path)

    grouped_data = eeg_data.groupby("WORD")
    list_eegs = []
    for word in list_words:
        da = grouped_data.get_group(word)
        da = da[~da['ELECNAME'].isin(["REJ1", "REJ2", "REJ3"])]  # .drop(columns=['WORD#', 'WORD', 'ELEC#', 'ELECNAME'])
        d = da.drop(columns=['WORD#', 'WORD', 'ELEC#', 'ELECNAME']).to_numpy()
        list_eegs.append(d)
    eeg_signals = np.stack(list_eegs)

    list_electrodes = pd.unique(eeg_data["ELECNAME"])[3:]

    dl_word_distances = None
    cosine_word_distances = None
    l2_word_distances = None

    if args.word_dist_repr == "levenshtein":
        dl_word_distances = compute_all_dl_distance(list_paired_words, normalize=True)
    elif args.word_dist_repr == "levenshtein_ipa":
        import eng_to_ipa as ipa
        list_ipa_words = [ipa.convert(word) for word in list_words]
        list_paired_ipa_words = all_pairs(list_ipa_words)
        dl_word_distances = compute_all_dl_distance(list_paired_ipa_words, normalize=True)
    else:
        if args.use_model_cache:
            # load Bert representations
            if args.word_dist_repr == "bert":
                word_features = np.load(os.path.join(args.save_folder, "kiloword_trained_bert_features.npy"))[all_ids]
            elif args.word_dist_repr == "bert_random":
                word_features = np.load(os.path.join(args.save_folder, "kiloword_random_bert_features.npy"))[all_ids]
            elif args.word_dist_repr == "hubert":
                word_features = np.load(os.path.join(args.save_folder, "kiloword_trained_hubert_features.npy"))[all_ids]
            elif args.word_dist_repr == "hubert_random":
                word_features = np.load(os.path.join(args.save_folder, "kiloword_random_hubert_features.npy"))[all_ids]
            else:
                word_features = np.load(os.path.join(args.save_folder, "kiloword_trained_bert_features.npy"))[all_ids]

        else:
            if "bert" in args.word_dist_repr:
                from transformers import BertTokenizer, BertConfig, BertModel
                tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                model = BertModel.from_pretrained("bert-base-uncased")
                word_features = get_model_representations(list_words, model, tokenizer)

        cosine_word_distances = compute_all_representations_distances(word_features.squeeze(1),
                                                                      list_paired_indices,
                                                                      norm="cosine")
        l2_word_distances = compute_all_representations_distances(word_features.squeeze(1),
                                                                  list_paired_indices)

    compute_correlations(eeg_signals,
                         cosine_word_distances,
                         l2_word_distances,
                         dl_word_distances,
                         list_paired_indices,
                         list_electrodes,
                         corr,
                         pad_step=args.pad_step,
                         timesteps=args.timesteps)

    print("DONE")


if __name__ == '__main__':
    from tqdm import tqdm
    from copy import deepcopy

    args = arg_parser()
    main(args)
    t_name = deepcopy(args.tab_name)
    # from tqdm import tqdm
    # for elt in tqdm(LIST_LABELS, total=len(LIST_LABELS), desc="Processing ..."):
    #     args.focus_label = elt
    #     args.tab_name = t_name
    #     args.word_dist_repr = "bert"
    #     main(args)
    #     args.tab_name = t_name
    #     args.focus_label = elt
    #     args.word_dist_repr = "bert_random"
    #     main(args)
    #     args.tab_name = t_name
    #     args.focus_label = elt
    #     args.word_dist_repr = "levenshtein"
    #     main(args)
    #     args.focus_label = elt
    #     args.word_dist_repr = "levenshtein_ipa"
    #     main(args)
