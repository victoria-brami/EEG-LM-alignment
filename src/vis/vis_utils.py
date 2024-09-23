import os

def build_destination_folder(table_path: str = "",
                             dataset_name: str = "",
                             save_folder: str = "",
                             distance_type: str = "l2",
                             timesteps: str = "31",
                             corr_type: str = "spearman"):
    label_name = os.path.basename(table_path).split("_")[-2]
    topo_name = os.path.basename(table_path).replace("csv", "png")
    if distance_type != "l2":
        topo_name = topo_name.replace(".png", f"_{distance_type}.png")

    if dataset_name not in save_folder:
        save_folder = os.path.join(save_folder, dataset_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

    if not os.path.exists(os.path.join(save_folder, "image")):
        os.mkdir(os.path.join(save_folder, "image"))

    dest_file_path = os.path.join(save_folder, "image", corr_type, distance_type, f"{timesteps}ms")

    if not os.path.exists(os.path.join(save_folder, "image", corr_type, distance_type, f"{timesteps}ms")):
        os.makedirs(os.path.join(save_folder, "image", corr_type, distance_type, f"{timesteps}ms"), exist_ok=True)

    return dest_file_path
