import os
import hydra
from omegaconf import DictConfig
from logging import getLogger
from src.vis import create_valid_gif


@hydra.main(config_path="../configs", config_name="animate")
def main(cfg: DictConfig) -> None:

    logger = getLogger(__name__)

    # Create the save path
    save_path = os.path.join(cfg.destpath, "image", cfg.corr, cfg.distance)
    os.makedirs(save_path, exist_ok=True)

    output_path = create_valid_gif(save_path, cfg.model.shortname, over_layers=True)
    logger.info(f"Animation saved to {output_path} ")


if __name__ == '__main__':
    main()
