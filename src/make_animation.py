import os
import hydra
from omegaconf import DictConfig
from logging import getLogger
from src.vis import create_valid_gif


@hydra.main(config_path="../configs", config_name="animate")
def main(cfg: DictConfig) -> None:

    logger = getLogger(__name__)

    cfg.destpath = os.path.join(cfg.destpath, cfg.distance, cfg.corr) #, f"{cfg.timesteps}ms")


    # Create the save path
    save_path = os.path.join(cfg.save_folder, cfg.data.dataname, cfg.label_name,
                             "image", cfg.corr, cfg.distance, f"{cfg.timesteps}ms")
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"loading from at {save_path}")

    print("model name", cfg.model.shortname)

    output_path = create_valid_gif(save_path, cfg.destpath, cfg.model.shortname, cfg.label_name, str(cfg.timesteps), over_layers=True)
    logger.info(f"Animation saved to {output_path} ")


if __name__ == '__main__':
    main()
