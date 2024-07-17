from typing import Optional, Union

from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class BaseDataset(Dataset):
    """
    Base class for all datasets (Abstract Base Class).
    """

    def __init__(self, config: Union[dict, OmegaConf], tokenizer: Optional[AutoTokenizer] = None):
        self.config = config
        self.datapath = config.datapath
        self.tokenizer = tokenizer
        self.filter_labels = config.labels

    def _load_channels(self):
        """ Method to load the electrodes names and 3d Positions"""
        raise NotImplementedError

    def _load_data(self):
        """ Method to load the EEG data and its corresponding labels"""
        raise NotImplementedError

    def _get_dataset_words(self):
        """ Method to get all the words paired to a given signal """
        raise NotImplementedError
