from argparse import ArgumentParser

import koroba.utils.io as io
from koroba.datamodules import SyntheticDataModule
from koroba.runner import Runner
from koroba.utils import Randomizer

from configs.synthetic_experiment_config import (
    CommonArguments,
    DataArguments,
    RunArguments,
    SpecificArguments,
)


def main(args):
    Randomizer.set_seed(seed=args.seed)

    datamodule = SyntheticDataModule(
        batch_size=args.batch_size,
        device=args.device,
        n_boxes=args.n_boxes,
        n_cameras=args.n_cameras,
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
        projection_mode=args.projection_mode,
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

