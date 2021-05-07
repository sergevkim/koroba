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
    angle_threshold: float = 0.3,
    batch_size: int = 1
    center_std: float = 0.2,
    center_threshold: float = 0.02,
    class_probability: float = 0.1,
    drop_probability: float = 0.2,
    size_mean: float = 0.05,
    size_std: float = 0.02,
    size_threshold: float = 0.3,


@dataclass
class RunArguments:
    learning_rate: float = 0.01
    max_epoch: int = 200
    one_batch_overfit: int = 1
    optimizer_name: str = 'adam'


@dataclass
class SpecificArguments:
    specific: bool = False


print(CommonArguments.device)

