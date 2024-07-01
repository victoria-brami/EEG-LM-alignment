import os
from argparse import ArgumentParser
from kiloword.vis import create_valid_gif




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--label", default="OBJECT")
    parser.add_argument("--distance", default="cosine")
    parser.add_argument("--corr", default="pearson")
    parser.add_argument("--model", default="bert")
    args = parser.parse_args()


    save_path = os.path.join("/home/viki/Downloads/kiloword_correlations",
                             args.label, "image", args.corr, args.distance)

    create_valid_gif(save_path,  args.model)
