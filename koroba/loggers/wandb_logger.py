from typing import Any, Dict, Optional

import wandb
from torch import Tensor
from torch.nn import Module
from wandb.wandb_run import Run

from koroba.loggers import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(
            self,
            project: str,
        ):
        self.experiment = wandb.init(project=project)

    def watch(
            self,
            model: Module,
            log: str = 'gradients',
            log_freq: int = 1000,
        ):
        self.experiment.watch(
            model=model,
            log=log,
            log_freq=log_freq,
        )

    def log_metrics(
            self,
            metrics: Dict[str, Any],
            step: Optional[int] = None,
        ):
        if step is None:
            self.experiment.log(metrics)
        else:
            self.experiment.log({
                **metrics,
                'trainer/global_step': step,
            })


if __name__ == '__main__':
    logger = WandbLogger(
        project='test',
    )
    print(logger)

