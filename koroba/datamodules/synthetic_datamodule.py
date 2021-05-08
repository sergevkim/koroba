import numpy as np
import torch

import koroba.utils.io as io
from koroba.datamodules import BaseDataModule
from koroba.utils import SyntheticData as SynData


class SyntheticDataModule(BaseDataModule):
    def __init__(
            self,
            batch_size: int = 1,
            device: torch.device = torch.device('cpu'),
            n_boxes: int = 10,
            n_cameras: int = 10,
            n_classes: int = 10,
        ):
        self.batch_size = batch_size
        self.device = device
        self.n_boxes = n_boxes
        self.n_cameras = n_cameras
        self.n_classes = n_classes

    def setup(
            self,
            angle_threshold: float = 0.3,
            center_std: float = 0.2,
            center_threshold: float = 0.02,
            class_probability: float = 0.1,
            drop_probability: float = 0.2,
            size_mean: float = 0.05,
            size_std: float = 0.02,
            size_threshold: float = 0.3,
        ):
        self.true, self.seen = SynData.generate_box_dataset(
            n=self.n_cameras,
            n_boxes=self.n_boxes,
            n_classes=self.n_classes,
            center_std=center_std,
            size_mean=size_mean,
            size_std=size_std,
            class_probability=class_probability,
            drop_probability=drop_probability,
            center_threshold=center_threshold,
            size_threshold=size_threshold,
            angle_threshold=angle_threshold,
        )
        for i, box in enumerate(self.true['boxes']):
            io.write_bounding_box(
                filename=f'output/true_box_{i}.pcd',
                box=box,
            )
        cameras = SynData.generate_camera_dataset(
            n=self.n_cameras,
            angle_threshold=0.3,
        )
        self.seen['cameras'] = cameras
        SynData.update_box_dataset_with_cameras(
            seen=self.seen,
            proj=False,
        )


        initial_boxes = \
            np.concatenate(tuple(filter(lambda x: len(x), self.seen['boxes'])))
        center_mean = np.mean(initial_boxes[:, :3], axis=0)
        center_std = np.std(initial_boxes[:, :3], axis=0)
        size_mean = np.mean(initial_boxes[:, 3:-1], axis=0)
        size_std = np.std(initial_boxes[:, 3:-1], axis=0)

        to_concat = (
            np.random.normal(center_mean, center_std, (self.n_boxes, 3)),
            np.abs(np.random.normal(size_mean, size_std, (self.n_boxes, 3))),
            np.random.uniform(0.0, 2 * np.pi, (self.n_boxes, 1)),
        )
        initial_boxes = np.concatenate(to_concat, axis=1)
        initial_boxes[:, 3:-1] = np.log(initial_boxes[:, 3:-1])
        initial_boxes = \
            torch.tensor(initial_boxes, dtype=torch.float, device=self.device)
        optimized_boxes = initial_boxes.clone().detach()
        optimized_boxes.requires_grad = True

        initial_scores = np.random.random((self.n_boxes, self.n_classes + 1))
        initial_scores[:, -1] = 0.0
        initial_scores = \
            torch.tensor(initial_scores, dtype=torch.float, device=self.device)
        optimized_scores = initial_scores.clone().detach()
        optimized_scores.requires_grad = True

        for i in range(len(self.seen['boxes'])):
            self.seen['boxes'][i] = torch.tensor(
                self.seen['boxes'][i],
                dtype=torch.float,
                device=self.device,
            )
            self.seen['labels'][i] = torch.tensor(
                self.seen['labels'][i],
                dtype=torch.long,
                device=self.device,
            )

        self.optimized = {
            'boxes': optimized_boxes,
            'scores': optimized_scores,
        }

    def get_constants(self):
        constants = {
            'n_boxes': n_boxes,
            'n_cameras': n_cameras,
            'n_classes': n_classes,
        }

        return constants