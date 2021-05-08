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

    datamodule = SyntheticDataModule(
        batch_size=args.batch_size,
        device=args.device,
        n_boxes=args.n_boxes,
        n_classes=args.n_classes,
    )
    datamodule.setup(
        center_std=args.center_std,
        size_mean=args.size_mean,
        size_std=args.size_std,
        class_probability=args.class_probability,
        drop_probability=args.drop_probability,
        center_threshold=args.center_threshold,
        size_threshold=args.size_threshold,
        angle_threshold=args.angle_threshold,
    )

    runner = Runner(
        device=args.device,
        max_epoch=args.max_epoch,
        optimizer_name=args.optimizer_name,
        verbose=args.verbose,
    )
    optimized_result = runner.run(
        datamodule=datamodule,
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

