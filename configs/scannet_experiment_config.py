from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class CommonArguments:
    device: torch.device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    project: str = 'koroba'
    seed: int = 9
    verbose: int = 1
    version: str = 'koroba0.1'


@dataclass
class DataArguments:
    batch_size: int = 1
    scan_path: Path = Path('./data/scans/scene0000_00')
    n_frames: int = 20


@dataclass
class RunArguments:
    learning_rate: float = 0.01
    max_epoch: int = 200
    mode: str = '2d'
    optimizer_name: str = 'adam'
    projection_mode: str = 'minmax'


@dataclass
class SpecificArguments:
    giou_coef: float = 0.5
    nll_coef: float = 0.5
    l1_coef: float = 0.0
    no_object_coef: float = 0.4


print(CommonArguments.device)

