import argparse
from kiloword.vis import plot_2d_topomap
from kiloword.evaluation import correlations



def vis_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_folder", type=str,
                        default="/home/viki/Downloads/kiloword_correlations",
                        help="folder where the experiments are saved")
    return parser.parse_args()


def main(args):
    pass


if __name__ == '__main__':
    args = vis_parser()
    main(args)