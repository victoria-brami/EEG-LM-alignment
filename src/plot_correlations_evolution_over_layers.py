import os

import hydra
import numpy as np

from src.dataset import get_dataset_electrodes, project_3d_coordinates_in_plan
from src.evaluation import CorrelationsTable
from src.utils import (
    extract_correlations_and_periods,
    split_into_chunks
)
from src.vis import build_destination_folder
from src.vis import plot_2d_topomap


@hydra.main(config_path='../configs', config_name='plot')
def main(config):
    modelname = config.model.shortname

    # Load the dataset electrodes locations and names
    data_path_name = "eeg_POS" if config.data.dataname == "ubira" else config.data.dataname
    electrodes = get_dataset_electrodes(config.data.rootpath, data_path_name)
    list_electrodes = electrodes["#NAME"].tolist()
    electrodes_pos = electrodes[["X", "Y", "Z"]].to_numpy()
    electrodes_pos = project_3d_coordinates_in_plan(electrodes_pos)

    head_radius = np.linalg.norm(electrodes_pos[26 , :2] - electrodes_pos[ 4, :2]) \
        if config.data.dataname == "ubira" else np.linalg.norm(electrodes_pos[26 , :2] - electrodes_pos[6, :2])

    # Get the src folder where the correlations tables are stored
    corr_save_folder = os.path.join(config.save_folder, "csv")

    # Build the destination path/folder

    # Load the list of correlations tables corresponding to each layer
    # of the chosen LM (take care of ordering them correctly)
    list_corr_names = [filename for filename in os.listdir(corr_save_folder)
                       if modelname in filename and "random" not in filename and "layer" in filename]

    reordered_list_corr_names = [(int(name.split(f"layer_")[1].split("_")[0]), name) for name in list_corr_names]
    reordered_list_corr_names.sort()
    list_corr_names = [elt[1] for elt in reordered_list_corr_names]

    print("\n CORRELARIONS NAMES", list_corr_names)

    list_corr_tabs = [CorrelationsTable(name=tab_name,
                                        table_folder=corr_save_folder,
                                        table_columns=config.tab_attrs,
                                        eval=True) for tab_name in list_corr_names]
    print(list_corr_tabs[0].name)

    # Re-order the table by time-wise
    list_results_grouped_tabs = [corr.extract_sub_table(attribute="distance", value=config.distance,
                                                        groupby_key="truncate_start") for corr in list_corr_tabs]

    # Extract the correlations
    pears_corr_values, spear_corr_values, sub_titles = [], [], []
    fig_sub_titles = []
    for layer_id in range(len(list_results_grouped_tabs)):
        pears_corr_val, spear_corr_val, sub_title = extract_correlations_and_periods(
            list_results_grouped_tabs[layer_id])
        pears_corr_values.append(pears_corr_val)
        # print("Layer ID ", layer_id, "pears", len(pears_corr_val), len(pears_corr_val[0]))
        # print("Layer ID", layer_id,  "spear", len(spear_corr_val), len(spear_corr_val[0]))
        spear_corr_values.append(spear_corr_val)
        if layer_id == 0:
            fig_sub_titles.extend(sub_title)
        sub_titles.append(f"{modelname}_layer_{layer_id}")

    # Permute the time window and the layers
    pears_corr_values = np.transpose(pears_corr_values, (1, 0, 2))
    spear_corr_values = np.transpose(spear_corr_values, (1, 0, 2))

    # Reshape the plots
    # print("subtitles", sub_titles)
    sub_titles = split_into_chunks(sub_titles, config.vis.chunk_size)
    # print("subtitles APRES", sub_titles)

    # Plot correlations topographies

    pears_dest_file_path = build_destination_folder(list_corr_tabs[-1].table_path, config.save_folder,
                                                    config.distance, "pearson")
    spear_dest_file_path = build_destination_folder(list_corr_tabs[-1].table_path, config.save_folder,
                                                    config.distance, "spearman")

    n_rows, n_cols = len(sub_titles), len(sub_titles[0])
    # print(n_rows, n_cols, len(sub_titles), len(pears_corr_values[0][0]))

    for subtitle_id in range(len(fig_sub_titles)):

        # Reshape the plots
        pears_corr_val = split_into_chunks(pears_corr_values[subtitle_id], config.vis.chunk_size)
        spear_corr_val = split_into_chunks(spear_corr_values[subtitle_id], config.vis.chunk_size)

        plot_2d_topomap(electrodes_pos[:, :2], pears_corr_val, dataname=config.data.dataname,
                        grid_res=config.vis.grid_res, head_radius=head_radius,
                        rows=n_rows, size=config.vis.size, cols=n_cols, edgecolor=config.vis.edgecolor,
                        subfig_name=sub_titles,
                        coords_name=list_electrodes, dpi=config.vis.dpi, title=fig_sub_titles[subtitle_id],
                        savepath=pears_dest_file_path + f"/pearson_{modelname}_{fig_sub_titles[subtitle_id]}.png")

        plot_2d_topomap(electrodes_pos[:, :2], spear_corr_val, dataname=config.data.dataname,
                        grid_res=config.vis.grid_res, head_radius=head_radius,
                        rows=n_rows, size=config.vis.size, cols=n_cols, edgecolor=config.vis.edgecolor,
                        subfig_name=sub_titles,
                        coords_name=list_electrodes, dpi=config.vis.dpi, title=fig_sub_titles[subtitle_id],
                        savepath=spear_dest_file_path + f"/spearman_{modelname}_{fig_sub_titles[subtitle_id]}.png")
    print("\033[96m JOB Done ! \033[0m")

if __name__ == '__main__':
    main()
