import pandas as pd
import numpy as np
from typing import Union
import torch
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from transformers import PreTrainedModel
from kiloword.utils.utils import normalize_data
from sklearn.cluster import KMeans
from pyxdameraulevenshtein import (
    damerau_levenshtein_distance,
    normalized_damerau_levenshtein_distance
)
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations


def extract_same_time_window_rows(window_start: int,
                                  window_end: int,
                                  tab: pd.DataFrame):
    extract_tab = tab[(tab["truncate_start"] == window_start) & (tab["truncate_end"] == window_end)]
    return extract_tab


def extract_same_electrode_rows(electrodes: Union[list, str],
                                tab: pd.DataFrame):
    if isinstance(electrodes, list):
        extract_tab = tab[tab["Channels"].isin(electrodes)]
    else:
        extract_tab = tab[tab["Channels"] == electrodes]
    return extract_tab


def get_representations(inputs: Union[list, torch.tensor],
                        model: PreTrainedModel) -> np.array:
    model.eval()
    hiddens = []

    with torch.no_grad():
        for i in tqdm(range(len(inputs))):
            if isinstance(inputs, dict):
                outputs = model(**inputs[i])
            else:
                outputs = model(inputs[i])
            hidden_states = outputs.last_hidden_state.cpu().numpy()
            hiddens.append(hidden_states[:, 0])
        return np.stack(hiddens)


def get_bert_representations(inputs: Union[list, torch.tensor],
                             model: PreTrainedModel) -> np.array:
    model.eval()
    hiddens = []

    with torch.no_grad():
        for i in tqdm(range(len(inputs))):
            outputs = model(**inputs[i])
            hidden_states = outputs.last_hidden_state.cpu().numpy()
            hiddens.append(hidden_states[:, 0])
        return np.stack(hiddens)


def compute_kmeans_labels(features: np.array,
                          n_clusters: int,
                          mode: str = "normal") -> np.array:
    features = normalize_data(features, mode=mode)

    kmeans = KMeans(n_clusters=n_clusters).fit(features)
    return kmeans.labels_


def all_pairs(elements):
    """

    :param elements:
    :return:
    """
    return np.array(list(combinations(elements, 2)))


def compute_all_dl_distance(list_pairs, normalize=True):
    """
    :param list_pairs:
    :param normalize:
    :return:
    """
    distances = []
    for (word1, word2) in tqdm(list_pairs):
        if normalize:
            dist = normalized_damerau_levenshtein_distance(word1, word2)
        else:
            dist = damerau_levenshtein_distance(word1, word2)
        distances.append(dist)
    return distances


def compute_all_representations_distances(features,
                                          list_paired_indices,
                                          norm="l2",
                                          normalize=True):
    distances = []
    for (id1) in tqdm(np.unique(list_paired_indices[:, 0])):
        id2 = list_paired_indices[np.where(np.array(list_paired_indices)[:, 0] == id1)[0], 1]
        if norm == "cosine":
            dist = cosine_similarity(features[id1][None, :], features[id2])[0]
        else:
            if normalize:
                norm1 = np.linalg.norm(features[id1])
                norm2 = np.linalg.norm(features[id2], axis=1)[:, None]
                feat1 = np.repeat(features[id1][None, :], len(id2), axis=0) / norm1
                feat2 = features[id2] / norm2
                dist = np.linalg.norm(feat1 - feat2, axis=1)
            else:
                dist = np.linalg.norm(np.repeat(features[id1][None, :], len(id2), axis=0) - features[id2], axis=1)
        distances.extend(dist)
    return distances


def compute_correlations(eegs,
                         cosine_word_distances,
                         l2_word_distances,
                         dl_distances,
                         list_paired_indices,
                         list_electrodes,
                         corr_table,
                         pad_step=10,
                         timesteps=31):

    for start_trunc in range(0, eegs.shape[-1] - timesteps + 1, pad_step):
        period = range(start_trunc, start_trunc + timesteps)

        # Restrict to channel-level analysis
        for chan_id in range(len(list_electrodes)):
            if isinstance(chan_id, int):
                ext_eegs = eegs[:, [chan_id], period]
                left_ch = [chan_id]
            else:
                ext_eegs = eegs[:, chan_id]
                ext_eegs = ext_eegs[:, :, period]
            if len(ext_eegs.shape) == 2:
                ext_eegs = ext_eegs[:, None, :]

            reshaped_eegs = ext_eegs.reshape(-1, ext_eegs.shape[1] * ext_eegs.shape[2])

            save_chan_id = chan_id
            if not isinstance(chan_id, list):
                save_chan_id = [save_chan_id]
            save_chan_id = '_'.join([list_electrodes[ch] for ch in save_chan_id])
            row_dict = {"Channel": f"Channel {save_chan_id} ",
                        "distance": None,
                        "truncate_start": period[0],
                        "truncate_end": period[-1],
                        "pearson": None,
                        "spearman": None}

            # reshaped_eegs = reegs.mean(1)
            cosine_distances = compute_all_representations_distances(reshaped_eegs, list_paired_indices,  norm="cosine")
            l2_distances = compute_all_representations_distances(reshaped_eegs, list_paired_indices)

            if cosine_word_distances is not None:
                pears_cos, _ = pearsonr(cosine_word_distances, cosine_distances)
                spear_cos, _ = spearmanr(cosine_word_distances, cosine_distances)

                row_dict["distance"] = "cosine"
                row_dict["pearson"] = pears_cos
                row_dict["spearman"] = spear_cos
                corr_table.update_table(row_dict)


            if l2_word_distances is not None:
                pears_l2, _ = pearsonr(l2_word_distances, l2_distances)
                spear_l2, _ = spearmanr(l2_word_distances, l2_distances)

                row_dict["distance"] = "l2"
                row_dict["pearson"] = pears_l2
                row_dict["spearman"] = spear_l2
                corr_table.update_table(row_dict)

            if dl_distances is not None:
                pears_dl, _ = pearsonr(dl_distances, l2_distances)
                spear_dl, _ = spearmanr(dl_distances, l2_distances)

                row_dict["distance"] = "levenshtein-l2"
                row_dict["pearson"] = pears_dl
                row_dict["spearman"] = spear_dl
                corr_table.update_table(row_dict)

                pears_dl_cos, _ = pearsonr(dl_distances, cosine_distances)
                spear_dl_cos, _ = spearmanr(dl_distances, cosine_distances)

                row_dict["distance"] = "levenshtein-cosine"
                row_dict["pearson"] = pears_dl_cos
                row_dict["spearman"] = spear_dl_cos

                corr_table.update_table(row_dict)

            corr_table.save_table()

