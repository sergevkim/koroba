from koroba.datamodules import SyntheticDataModule
from koroba.runner import Runner
from koroba.utils import Randomizer

from configs.run_synthetic_config import (
    CommonArguments,
    DataArguments,
    RunArhuments,
    SpecificArguments,
)


def main(args):
    Randomizer.set_seed(seed=args.seed)

    datamodule = SyntheticDataModule()
    datamodule.setup()

    runner = Runner(
        device=args.device,
        max_epoch=args.max_epoch,
        verbose=args.verbose,
    )
    runner.run(
        datamodule=datamodule,
        optimizer_name=args.optimizer_name,
    )


if __name__ == '__main__':
    parser = ArgumentParser()
    default_args_dict = {
        **vars(CommonArguments()),
        **vars(DataArguments()),
        **vars(RunArguments()),
        **vars(SpecificArguments()),
    }

    for arg, value in default_args_dict.items():
        parser.add_argument(
            f'--{arg}',
            type=type(value),
            default=value,
            help=f'<{arg}>, default: {value}',
        )

    args = parser.parse_args()
    main(args)

