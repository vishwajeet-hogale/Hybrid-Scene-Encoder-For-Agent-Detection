import hydra
from omegaconf import DictConfig
import lightning.pytorch as L


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
