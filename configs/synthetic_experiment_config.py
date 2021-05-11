from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    seed: int = 9
    verbose: int = 1
    version: str = 'koroba0.1'


@dataclass
class DataArguments:
    angle_threshold: float = 0.3
    batch_size: int = 1
    center_std: float = 0.2
    center_threshold: float = 0.02
    class_probability: float = 0.1
    drop_probability: float = 0.2
    size_mean: float = 0.05
    size_std: float = 0.02
    size_threshold: float = 0.1


@dataclass
class RunArguments:
    learning_rate: float = 0.01
    max_epoch: int = 200
    mode: str = '3d'
    one_batch_overfit: int = 1
    optimizer_name: str = 'adam'
    projection_mode: str = 'minmax'


@dataclass
class SpecificArguments:
    n_boxes: int = 8
    n_cameras: int = 20
    n_classes: int = 10
    giou_coef: float = 0.5
    nll_coef: float = 0.5
    l1_coef: float = 0.0
    no_object_coef: float = 0.4


print(CommonArguments.device)

