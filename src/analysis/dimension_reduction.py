from typing import Union
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from src.utils.utils import normalize_data


def compute_tsne(data: Union[list, np.array],
                 normalize: str="min_max",
                 **tsne_args):
    if normalize is not None:
        data = normalize_data(data, mode=normalize)
    tsne = TSNE(**tsne_args)
    features = tsne.fit_transform(data)
    return features

def compute_pca(data: Union[list, np.array],
                normalize: str="min_max",
                **pca_args):
    if normalize is not None:
        data = normalize_data(data, mode=normalize)
    pca = PCA(**pca_args)
    features = pca.fit_transform(data)
    return features
