import os
import hydra
import torch
import numpy as np
from src.dataset import get_dataset
from src.analysis import (
    all_pairs,
    get_model_representations,
    compute_all_representations_distances,
    compute_all_dl_distance,
    compute_correlations,
    get_model
)
from src.evaluation import CorrelationsTable



@hydra.main(config_path='../configs', config_name='correlations')
def main(config):

    # Initialize the dataset

    model, tokenizer = get_model(config.model.name)
    dataset = get_dataset(config.data, tokenizer, config.data.dataname)

    # Initialize the language model used
    list_words = [dataset[i]["word"] for i in range(len(dataset))] # getattr(dataset, "list_words")
    list_paired_words = all_pairs(list_words)
    list_paired_indices = all_pairs(range(len(list_words)))

    # Initialize the Correlations' table
    # Initialize Experiment table
    if config.model.layer != -1:
        config.tab_name = config.tab_name.replace(config.model.shortname, f"{config.model.shortname}_layer_{config.model.layer}")
    corr_save_folder = config.save_folder
    corr = CorrelationsTable(name=config.tab_name,
                             table_folder=corr_save_folder,
                             table_columns=config.tab_attrs)


    dl_word_distances = None


    if config.word_distance == "levenshtein":
        dl_word_distances = compute_all_dl_distance(list_paired_words, normalize=True)
        cosine_word_distances = None
        l2_word_distances = None
    elif config.word_distance == "levenshtein_ipa":
        import eng_to_ipa as ipa
        list_ipa_words = [ipa.convert(word) for word in list_words]
        list_paired_ipa_words = all_pairs(list_ipa_words)
        dl_word_distances = compute_all_dl_distance(list_paired_ipa_words, normalize=True)
        cosine_word_distances = None
        l2_word_distances = None

    else:
        config.model.layer = int(config.model.layer)
        word_features = get_model_representations(list_words, model, config.model.layer, tokenizer)

        if len(word_features.shape) == 3:
            word_features = word_features.squeeze(1)

        cosine_word_distances = compute_all_representations_distances(word_features,
                                                                      list_paired_indices,
                                                                      norm="cosine")
        l2_word_distances = compute_all_representations_distances(word_features,
                                                                  list_paired_indices)
    eeg_signals = np.stack([dataset[i]["raw_eeg_input_ids"] for i in range(len(dataset))])
    list_electrodes = dataset.channels["#NAME"]

    compute_correlations(eeg_signals,
                         cosine_word_distances,
                         l2_word_distances,
                         dl_word_distances,
                         list_paired_indices,
                         list_electrodes,
                         corr,
                         pad_step=config.pad_step,
                         timesteps=config.timesteps)

    print("Correlations Computed !")


if __name__ == '__main__':
    main()