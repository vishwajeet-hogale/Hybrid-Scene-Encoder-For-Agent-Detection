import os

# One BLAS/OpenMP thread per process (set before numpy/torch load; DataLoader workers inherit it). Each worker
# otherwise starts its own BLAS thread pool for small numpy matmuls; under memory pressure those pools can
# busy-wait for minutes inside dgemm and stall training.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import hydra
import torch
from omegaconf import DictConfig
import lightning.pytorch as L


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed, workers=True)

    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer)

    # fine-tuning: start from trained weights with a fresh optimizer / LR schedule (unlike ckpt_path resume)
    if cfg.get("init_ckpt"):
        state = torch.load(hydra.utils.to_absolute_path(cfg.init_ckpt), map_location="cpu", weights_only=False)
        missing = model.load_pretrained(state["state_dict"])
        print(f"Initialized weights from {cfg.init_ckpt} ({len(missing)} parameters newly initialized)")

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
