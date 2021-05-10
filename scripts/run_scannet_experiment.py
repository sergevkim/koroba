from argparse import ArgumentParser

import koroba.utils.io as io
from koroba.datamodules import ScanNetDataModule
from koroba.runner import Runner
from koroba.utils import Randomizer

from configs.scannet_experiment_config import (
    CommonArguments,
    DataArguments,
    RunArguments,
    SpecificArguments,
)


def main(args):
    Randomizer.set_seed(seed=args.seed)

    print('!',args.scan_path)
    print('!',type(args.scan_path))
    datamodule = ScanNetDataModule(
        batch_size=args.batch_size,
        scan_path=args.scan_path,
        device=args.device,
    )
    datamodule.setup()

    runner = Runner(
        device=args.device,
        max_epoch=args.max_epoch,
        optimizer_name=args.optimizer_name,
        box_matching_mode=args.box_matching_mode,
        giou_coef=args.giou_coef,
        nll_coef=args.nll_coef,
        l1_coef=args.l1_coef,
        no_object_coef=args.no_object_coef,
        verbose=args.verbose,
    )
    optimized_result = runner.run(
        datamodule=datamodule,
        mode=args.mode,
    )

    optimized_boxes = optimized_result['boxes']

    for i, box in enumerate(optimized_boxes):
        io.write_bounding_box(
            filename=f'output/optimized_box_{i}.pcd',
            box=box,
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

