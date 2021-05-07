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
    batch_size: int = 1
    data_path: Path = Path('./data')


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

