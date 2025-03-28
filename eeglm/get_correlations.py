import argparse
import os.path
import re
from copy import deepcopy
from logging import getLogger

import numpy as np
import pandas as pd

from eeglm.analysis import (
    all_pairs,
    compute_all_dl_distance,
    compute_all_representations_distances,
    compute_correlations,
    get_model_representations,
)
from eeglm.config import Config as cfg
from eeglm.evaluation import CorrelationsTable
from eeglm.utils import get_model, parse_table_labels, read_table

LIST_LABELS = [
    "SEPARATION",
    "LOCATION",
    "ENTERTAINMENT",
    "MONEY",
    "NATURE",
    "QUANTITY",
    "POLITICS",
    "RELIGION",
    "HOUSE",
    "MOVE",
    "SPORT",
    "JUSTICE",
    "INDUSTRY",
    "LANGUAGE",
    "FOOD",
    "MODE",
    "DEVICE",
    "FAMILY",
    "MUSIC",
    "CRIME",
    "CATASTROPHE",
    "ARMY",
    "TIME",
    "SCHOOL",
    "CLEANNESS",
    "DEATH",
    "GLORY",
    "BODY",
    "PEOPLE",
    "MEDICAL",
    "MATERIAL",
    "GOVERN",
    "SCIENCE",
    "PHILOSOPHY",
    "FEELING",
]

logger = getLogger(__name__)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--focus_label",
        type=str,
        default=None,  # choices=LIST_LABELS,
        help="compute correlations of a specific semantic field",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="./data/kiloword",
        help="folder where the experiments are saved",
    )
    parser.add_argument(
        "--tab_name",
        type=str,
        default="correlations.csv",
        help="Name of the document where the experiments are saved",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default=cfg.LABELS_PATH,
        help="File containing the annotations",
    )
    parser.add_argument(
        "--use_model_cache",
        action="store_true",
        help="whether to load pre-computed word representations or not",
    )
    parser.add_argument(
        "--eeg_path",
        type=str,
        default=cfg.DATA,
        help="File containing the EEG recordings",
    )
    parser.add_argument(
        "--word_dist_repr_old",
        type=str,
        default="bert",
        choices=[
            "bert",
            "bert_random",
            *[f"bert_layer_{i}" for i in range(12)],
            "canine_s",
            "canine_c",
            "canine_c_random",
            "canine_s_random",
            *[f"canine_s_layer_{i+1}" for i in range(16)],
            *[f"canine_c_layer_{i+1}" for i in range(16)],
            "hubert",
            "hubert_random",
            "bart",
            "bart_random",
            "levenshtein",
            "levenshtein_ipa",
        ],
        help="Word representations type",
    )

    parser.add_argument("--layer", type=int, default=-1, help="")

    parser.add_argument(
        "--word_dist_repr",
        type=str,
        default="bert-base-uncased",
        # choices=[
        #     "bert-base-uncased",
        #     "meta-llama/Llama-3.2-1B",
        # ],
        help="Word representations type",
    )
    parser.add_argument(
        "--use_random",
        action="store_true",
        help="Word representations type",
    )
    parser.add_argument(
        "--tab_attrs",
        type=list,
        nargs="+",
        default=[
            "Channel",
            "distance",
            "truncate_start",
            "truncate_end",
            "pearson",
            "spearman",
        ],
        help="keys to figure in the saved documents",
    )
    parser.add_argument("--pad_step", type=int, default=10, help="padding step")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=31,
        help="duration of the eeg signals extracted",
    )
    return parser.parse_args()


def main(args):

    logger.info(f"Using model: {re.sub(r'\.\./|/', '', args.word_dist_repr)}")
    # Download the labels
    labels = read_table(args.labels_path)
    labels_table = parse_table_labels(
        labels, LIST_LABELS, labelcolname="SEMANTIC_FIELD"
    )

    all_ids = np.arange(len(labels))

    if args.focus_label is not None and args.focus_label != "none":
        if args.focus_label == "OBJECT":
            all_ids = labels[labels["MATERIAL"] == "YES"].index
        elif args.focus_label == "ABSTRACT":
            all_ids = labels[labels["MATERIAL"] != "YES"].index
        else:
            all_ids = labels_table[labels_table[args.focus_label] == True].index
        args.tab_name = "_".join([args.focus_label, args.tab_name])
    else:
        args.tab_name = "_".join(["ALL", args.tab_name])

    rep_name = re.sub(r"\.\./|/", "", args.word_dist_repr)
    if args.use_random:
        rep_name = "random_" + rep_name
    if args.layer != -1:
        rep_name += f"_layer_{args.layer}"

    args.tab_name = "_".join([rep_name, str(args.timesteps) + "ms", args.tab_name])
    # Get the list of words and their pairs
    list_words = labels["WORD"].values[all_ids]
    list_paired_words = all_pairs(list_words)
    list_paired_indices = all_pairs(range(len(list_words)))

    if args.focus_label is not None:
        corr_save_folder = os.path.join(args.save_folder, args.focus_label)
        os.makedirs(corr_save_folder, exist_ok=True)
    else:
        args.focus_label = "ALL"
        corr_save_folder = os.path.join(args.save_folder, args.focus_label)
        os.makedirs(corr_save_folder, exist_ok=True)
    corr_save_folder = os.path.join(corr_save_folder, "csv")
    os.makedirs(corr_save_folder, exist_ok=True)

    # Initialize Experiment table
    corr = CorrelationsTable(
        name=args.tab_name, table_folder=corr_save_folder, table_columns=args.tab_attrs
    )

    # Download the EEG data and drop non-useful info
    eeg_data = read_table(args.eeg_path)

    grouped_data = eeg_data.groupby("WORD")
    list_eegs = []
    for word in list_words:
        da = grouped_data.get_group(word)
        da = da[~da["ELECNAME"].isin(["REJ1", "REJ2", "REJ3"])]
        d = da.drop(columns=["WORD#", "WORD", "ELEC#", "ELECNAME"]).to_numpy()
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
        dl_word_distances = compute_all_dl_distance(
            list_paired_ipa_words, normalize=True
        )
    else:
        if args.use_model_cache:
            if "random" in args.word_dist_repr:
                word_features = np.load(
                    os.path.join(
                        args.save_folder,
                        "word_features",
                        f"kiloword_random_{args.word_dist_repr.split('_random')[0].split('random_')[0]}_features.npy",
                    )
                )[all_ids]
            else:
                word_features = np.load(
                    os.path.join(
                        args.save_folder,
                        "word_features",
                        f"kiloword_trained_{args.word_dist_repr}_features.npy",
                    )
                )[all_ids]

        else:

            model, tokenizer = get_model(args.word_dist_repr, args.use_random)

            word_features = get_model_representations(
                list_words, model, layer=args.layer, tokenizer=tokenizer
            )

        print("Number of words ", word_features.shape[0])
        if len(word_features.shape) == 3:
            word_features = word_features.squeeze(1)
        cosine_word_distances = compute_all_representations_distances(
            word_features, list_paired_indices, norm="cosine"
        )
        l2_word_distances = compute_all_representations_distances(
            word_features, list_paired_indices
        )

    compute_correlations(
        eeg_signals,
        cosine_word_distances,
        l2_word_distances,
        dl_word_distances,
        list_paired_indices,
        list_electrodes,
        corr,
        pad_step=args.pad_step,
        timesteps=args.timesteps,
    )

    print("DONE")


if __name__ == "__main__":

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
