import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import tqdm

from koroba.datamodules import BaseDataModule
from koroba.utils import Box


class ScanNetDataModule(BaseDataModule):
    def __init__(
            self,
            batch_size: int = 1,
            device: torch.device = torch.device('cpu'),
            scan_path: Path = Path('./data/scans/scene0000_00'),
        ):
        self.batch_size = batch_size
        self.device = device
        self.scan_path = scan_path

    def handle_one_object_on_frame(
            self,
            frame,
            object_idx: int,
        ):
        mask = frame == object_idx
        mask = mask[:, :, 0]
        mask = np.pad(
            mask,
            pad_width=(1, 1),
            mode='constant',
            constant_values=(0, 0),
        )

        if mask.sum() == 0:
            return None

        #bounding box
        mask_x1 = (mask.argmax(axis=0) != 0)[::-1]
        x_max = len(mask_x1) - mask_x1.argmax() - 1 - 1
        x_min = (mask.argmax(axis=0) != 0).argmax() - 1

        mask_y1 = (mask.argmax(axis=1) != 0)[::-1]
        y_max = len(mask_y1) - mask_y1.argmax() - 1 - 1
        y_min = (mask.argmax(axis=1) != 0).argmax() - 1

        vertices = torch.tensor(
            ((x_min, y_min), (x_max, y_max)),
            dtype=torch.float,
            device=self.device,
        )
        box = Box.vertices2d_to_box2d(vertices=vertices)

        return box

    def prepare_frame_info(
            self,
            frame_path: Path,
        ):
        projections_set = list()
        labels = list()

        for object_idx in range(1, self.n_boxes + 1):
            frame = cv2.imread(str(frame_path))
            projection = self.handle_one_object_on_frame(
                frame=frame,
                object_idx=object_idx,
            )

            if projection is not None:
                projections_set.append(projection)
                labels.append(object_idx)

        projections_set = torch.stack(projections_set, dim=0)
        labels = torch.tensor(
            labels,
            dtype=torch.float,
            device=self.device,
        )
        scores = torch.ones(len(projections_set))

        info = {
            'projections_set': projections_set,
            'labels': labels,
            'scores': scores,
        }

        return info

    def prepare_seen(
            self,
            n_frames: int,
        ):
        seen = defaultdict(list)

        aggregation_info_path = \
            self.scan_path / f'{self.scan_path.name}.aggregation.json'
        instance_path = self.scan_path / 'instance-filt'
        frames_paths = [p for p in instance_path.glob('*.png')]
        poses_path = self.scan_path / 'sens_info/pose'
        poses_paths = [p for p in poses_path.glob('*.txt')]
        intrinsic_path = \
            self.scan_path / 'sens_info/intrinsic/intrinsic_color.txt'

        intrinsic_matrix = np.loadtxt(intrinsic_path)[:, :-1]
        intrinsic = torch.tensor(intrinsic_matrix, device=self.device)

        assert len(poses_paths) == len(frames_paths)

        with open(aggregation_info_path) as json_file:
            aggregation_info = json.load(json_file)
            self.n_boxes = len(aggregation_info['segGroups'])

        self.n_cameras = len(poses_paths)

        for i in tqdm.tqdm(range(min(len(frames_paths), n_frames))):
            frame_info = self.prepare_frame_info(frame_path=frames_paths[i])
            pose_matrix = np.loadtxt(poses_paths[i])
            pose = torch.tensor(pose_matrix, device=self.device)
            camera = intrinsic @ pose
            seen['projections_sets'].append(frame_info['projections_set'])
            seen['labels'].append(frame_info['labels'])
            seen['scores'].append(frame_info['scores'])
            seen['camera'].append(camera)

        return seen

    def prepare_optimized(self):
        initial_boxes = torch.cat(
            tuple(filter(lambda x: len(x), self.seen['projections_sets'])),
            dim=0,
        )

        xy_center_mean = initial_boxes[:, :2].mean(axis=0)
        print(xy_center_mean, xy_center_mean.mean())
        z_center_mean = torch.tensor(
            [xy_center_mean.mean()],
            device=self.device,
        )
        center_mean = torch.cat((xy_center_mean, z_center_mean), dim=0)

        xy_center_std = initial_boxes[:, :2].std(axis=0)
        z_center_std = torch.tensor(
            [xy_center_std.mean()],
            device=self.device,
        )
        center_std = torch.cat((xy_center_std, z_center_std), dim=0)

        xy_size_mean = initial_boxes[:, 2:-1].mean(axis=0)
        z_size_mean = torch.tensor(
            [xy_size_mean.mean()],
            device=self.device,
        )
        size_mean = torch.cat((xy_size_mean, z_size_mean), dim=0)

        xy_size_std = initial_boxes[:, 2:-1].std(axis=0)
        z_size_std = torch.tensor(
            [xy_size_std.mean()],
            device=self.device,
        )
        size_std = torch.cat((xy_size_std, z_size_std), dim=0)

        to_concat = (
            torch.tensor(
                np.random.normal(
                    center_mean.cpu().numpy(),
                    center_std.cpu().numpy(),
                    (self.n_boxes, 3),
                ),
                dtype=torch.float,
            ),
            torch.tensor(
                np.abs(np.random.normal(
                    size_mean.cpu().numpy(),
                    size_std.cpu().numpy(),
                    (self.n_boxes, 3),
                )),
                dtype=torch.float,
            ),
            torch.tensor(
                np.random.uniform(0.0, 2 * np.pi, (self.n_boxes, 1)),
                dtype=torch.float,
            ),
        )
        initial_boxes = torch.cat(to_concat, axis=1)
        initial_boxes = \
            torch.tensor(initial_boxes, dtype=torch.float, device=self.device)
        optimized_boxes = initial_boxes.clone().detach()
        optimized_boxes.requires_grad = True

        initial_scores = np.random.random((self.n_boxes, self.n_boxes + 1))
        initial_scores[:, -1] = 0.0
        initial_scores = \
            torch.tensor(initial_scores, dtype=torch.float, device=self.device)
        optimized_scores = initial_scores.clone().detach()
        optimized_scores.requires_grad = True

        optimized = {
            'boxes': optimized_boxes,
            'scores': optimized_scores,
        }

        return optimized

    def setup(
            self,
            n_frames: int = 100,
        ):
        self.true = None
        self.seen = self.prepare_seen(n_frames=n_frames)
        self.optimized = self.prepare_optimized()

    def get_constants(self):
        constants = {
            'n_boxes': self.n_boxes,
            'n_cameras': self.n_cameras,
            'n_classes': self.n_boxes,
        }

        return constants
