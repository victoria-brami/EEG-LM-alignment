import pandas as pd
import numpy as np
from typing import Union
import torch
from tqdm import tqdm
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
                          mode: str="normal") -> np.array:

  features = normalize_data(features, mode=mode)

  kmeans = KMeans(n_clusters=n_clusters).fit(features)
  return kmeans.labels_

def all_pairs(elements):
    """

    :param elements:
    :return:
    """
    return list(combinations(elements, 2))

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

def compute_all_representations_distances2(features,
                                          list_paired_indices,
                                          norm="l2",
                                          normalize=True):
  distances = []
  for (id1) in tqdm(np.unique(list_paired_indices[:, 0])):
    id2 = list_paired_indices[np.where(np.array(list_paired_indices)[:, 0] == id1)[0], 1]
    if norm == "cosine":
      dist = cosine_similarity(features[id1][None, :], features[id2])
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