import numpy as np
import argparse
import os
import hydra

import pandas as pd

from src.vis import plot_2d_topomap
from src.utils import (
    read_table,
    extract_correlations_and_periods,
    split_into_chunks
)
from src.evaluation import CorrelationsTable
from src.vis import build_destination_folder
from src.dataset import get_dataset

def vis_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str,
                        default="/home/viki/Downloads/kiloword",
                        help="folder where the experiments are saved")
    parser.add_argument("--dataset_name", type=str, default="kiloword")
    parser.add_argument("--dataset_path", type=str,
                        default="/home/viki/mne_data/MNE-kiloword-data",
                        help="Path to where all the info of the dataset is stored")
    parser.add_argument("--tab_attrs", type=list,
                        default=['Channel', 'distance', 'truncate_start', 'truncate_end', 'pearson', 'spearman'],
                        nargs="+",
                        help="keys to figure in the saved documents")
    parser.add_argument("--model", type=str,
                        default="canine_s",
                        help="Name of the Language Model used")
    parser.add_argument("--label_name", type=str,
                        default="OBJECT",
                        help="Name of the Label used")
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=31,
                        help="duration of the eeg signals extracted")
    parser.add_argument("-d", "--distance", type=str,
                        default="l2",
                        choices=["cosine", "l2", "levenshtein-l2", "levenshtein-cosine"],
                        help="distance between EEG representations")
    return parser.parse_args()



def build_source_folder(table_path: str = "",
                           distance_type: str = "l2",
                           corr_type: str = "spearman"):
    pass


@hydra.main(config_path='../configs', config_name='visualize')
def main(config):
    # # Load the electrodes names
    # eeg_data = read_table(cfg.DATA)
    # list_electrodes = pd.unique(eeg_data["ELECNAME"])[3:]
    # # Load the electrodes coordinates
    # electrodes_pos = np.load(os.path.join(cfg.MNE_PATH, "locs3d.npy"))[:, :2]

    dataset = get_dataset(config.data, None, config.data.dataname)
    list_electrodes = dataset.channel_names
    electrodes_pos = dataset.channels

    if config.data.dataname == "harry_potter":
        list_electrodes = [list_electrodes[i] for i in range(0, len(list_electrodes), 3)]
    

    # Plot correlations topographies
    label_name = config.label_name
    save_folder = os.path.join(config.save_folder, config.data.dataname, label_name)

    # if not os.path.exists(os.path.join(args.save_folder, label_name)):
    #     os.mkdir(os.path.join(args.save_folder, label_name))

    corr_save_folder = os.path.join(save_folder, "csv")
    model = config.model.shortname
    print(model)
    print(os.listdir(corr_save_folder))

    list_corr_names = [filename for filename in os.listdir(corr_save_folder)
                       if model in filename and "random" not in filename and "layer" in filename and f"{config.timesteps}ms" in filename]
    print("names", len(list_corr_names))
    print("names", "\n ".join(list_corr_names))

    # TODO: Sort the correlation tables in the right order
    # assert len(list_corr_names) > 1, "Cannot plot that kind of figure for this model"
    reordered_list_corr_names = [(int(name.split(f"layer_")[1].split("_")[0]), name) for name in list_corr_names]
    reordered_list_corr_names.sort()
    list_corr_names = [elt[1] for elt in reordered_list_corr_names]

    # Get all tables over the model layers
    list_corr_tabs = [CorrelationsTable(name=tab_name,
                                 table_folder=corr_save_folder,
                                 table_columns=config.tab_attrs,
                                 eval=True) for tab_name in list_corr_names]




    # Re-order the table by time-wise
    list_results_grouped_tabs = [
        corr.extract_sub_table(attribute=["distance"],
                               value=[config.distance],
                               groupby_key="truncate_start") for corr in list_corr_tabs]

    # Extract the correlations
    pears_corr_values, spear_corr_values, sub_titles = [], [], []
    fig_sub_titles = []
    for layer_id in range(len(list_results_grouped_tabs)):
        pears_corr_val, spear_corr_val, sub_title = extract_correlations_and_periods(list_results_grouped_tabs[layer_id])
        pears_corr_values.append(pears_corr_val)
        spear_corr_values.append(spear_corr_val)
        if layer_id == 0:
            fig_sub_titles.extend(sub_title)
        sub_titles.append(f"{model}_layer_{layer_id}")


    # Permute the time window and the layers
    pears_corr_values = np.transpose(pears_corr_values, (1, 0, 2))
    spear_corr_values = np.transpose(spear_corr_values, (1, 0, 2))

    # Reshape the plots
    sub_titles = split_into_chunks(sub_titles,  config.chunk_size)

    # Plot correlations topographies

    pears_dest_file_path = build_destination_folder(list_corr_tabs[-1].table_path,
                                                    config.data.dataname,
                                                    config.save_folder,
                                                    config.distance,
                                                    config.timesteps,
                                                    "pearson")
    spear_dest_file_path = build_destination_folder(list_corr_tabs[-1].table_path,
                                                    config.data.dataname,
                                                    config.save_folder,
                                                    config.distance,
                                                    config.timesteps,
                                                    "spearman")

    # print("len Pearson", len(pears_corr_values), pears_corr_values)

    n_rows, n_cols = len(sub_titles), len(sub_titles[0])
    # print(n_rows, n_cols, len(sub_titles), len(pears_corr_values[0][0]))

    for subtitle_id in range(len(fig_sub_titles)):
        # Reshape the plots
        pears_corr_val = split_into_chunks(pears_corr_values[subtitle_id], config.chunk_size)
        spear_corr_val = split_into_chunks(spear_corr_values[subtitle_id], config.chunk_size)

        # print(len(pears_corr_val[-1]), len(spear_corr_val[-1]))

        plot_2d_topomap(electrodes_pos, pears_corr_val,
                        dataname=config.data.dataname,
                        grid_res=1000,
                        rows=n_rows, size=4, cols=n_cols, edgecolor="navy",
                        subfig_name=sub_titles,
                        coords_name=list_electrodes, dpi=200, title=fig_sub_titles[subtitle_id],
                        savepath=pears_dest_file_path + f"/pearson_{model}_{fig_sub_titles[subtitle_id]}.png")

        plot_2d_topomap(electrodes_pos, spear_corr_val,
                        dataname=config.data.dataname,
                        grid_res=100,
                        rows=n_rows, size=4, cols=n_cols, edgecolor="navy",
                        subfig_name=sub_titles,
                        coords_name=list_electrodes, dpi=200, title=fig_sub_titles[subtitle_id],
                        savepath=spear_dest_file_path + f"/spearman_{model}_{fig_sub_titles[subtitle_id]}.png")

    print(f"Done for label {config.label_name} !!!")


if __name__ == '__main__':
    from tqdm import tqdm
    # "OBJECT", "FEELING", "LOCATION", "PEOPLE",
    LABELS = ["OBJECT", "PEOPLE", "HOUSE", "MOVE", "FOOD",  "MODE", "DEVICE", "TIME",
             "DEATH", "BODY",  "MEDICAL",  "NATURE", "QUANTITY", "MATERIAL"]
    LABELS = ["MUSIC", "NATURE", "QUANTITY", "RELIGION", "DEATH", "HOUSE", "MOVE", "INDUSTRY", "TIME"]


    main()