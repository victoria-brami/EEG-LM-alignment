import os

def build_destination_folder(table_path: str = "",
                             save_folder: str = "",
                             distance_type: str = "l2",
                             corr_type: str = "spearman"):
    label_name = os.path.basename(table_path).split("_")[-2]
    topo_name = os.path.basename(table_path).replace("csv", "png")
    if distance_type != "l2":
        topo_name = topo_name.replace(".png", f"_{distance_type}.png")
    if not os.path.exists(os.path.join(save_folder, "image")):
        os.mkdir(os.path.join(save_folder, "image"))

    dest_file_path = os.path.join(save_folder, "image", corr_type, distance_type)

    if not os.path.exists(os.path.join(save_folder, "image", corr_type, distance_type)):
        os.makedirs(os.path.join(save_folder, "image", corr_type, distance_type), exist_ok=True)

    return dest_file_path
