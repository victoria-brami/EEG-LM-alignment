import os
from argparse import ArgumentParser
from kiloword.vis import create_valid_gif




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--label_name", default="OBJECT")
    parser.add_argument("--distance", default="cosine")
    parser.add_argument("--corr", default="pearson")
    parser.add_argument("--model", default="canine_s")
    parser.add_argument("--over_layers", default=True)
    args = parser.parse_args()

    LABELS = ["MONEY", "MUSIC", "NATURE", "QUANTITY",  "DEATH", "HOUSE", "MOVE", "RELIGION","INDUSTRY", "TIME"]

    for label in LABELS[3:]:
        args.label_name = label
        save_path = os.path.join("/home/viki/Downloads/kiloword_correlations",
                                 args.label_name, "image", args.corr, args.distance)

        create_valid_gif(save_path,  args.model, over_layers=args.over_layers)
