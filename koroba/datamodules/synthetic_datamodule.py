from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import koroba.utils.io as io
from koroba.datamodules import BaseDataModule
from koroba.utils import Camera


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

    def augment(
            self,
            x,
            m_threshold=None,
            a_threshold=None,
        ):
        if m_threshold is not None:
            x = x * np.random.uniform(
                1.0 - m_threshold,
                1.0 + m_threshold,
                x.shape,
            )
        if a_threshold is not None:
            x = x + np.random.uniform(-a_threshold, a_threshold, x.shape)

        return x

    def create_rotation_matrix(
            self,
            forward_vector,
        ):
        v1 = forward_vector / np.linalg.norm(forward_vector)
        v2 = np.cross([0.0, 0.0, 1.0], v1)
        v2 = v2 / np.linalg.norm(v2)
        v3 = np.cross(v1, v2)

        #three axes; v1 is collinear to the forward vect
        return np.stack((v2, v3, v1), axis=1)

    def generate_box_dataset(
            self,
            n,
            n_boxes,
            n_classes,
            center_std,
            size_mean,
            size_std,
            class_probability,
            drop_probability,
            center_threshold,
            size_threshold,
            angle_threshold,
        ):
        to_concat = (
            torch.tensor(
                np.random.normal(0.5, center_std, (n_boxes, 3)),
                dtype=torch.float,
            ),
            torch.tensor(
                np.abs(np.random.normal(size_mean, size_std, (n_boxes, 3))),
                dtype=torch.float,
            ),
            torch.tensor(
                np.random.uniform(0.0, 2 * np.pi, (n_boxes, 1)),
                dtype=torch.float,
            ),
        )
        boxes = torch.cat(to_concat, axis=1)
        true = {
            'boxes': boxes,
            'labels': np.random.choice(np.arange(n_classes), n_boxes)
        }

        seen = defaultdict(list)

        for _ in range(n):
            augmented_boxes = (
                self.augment(true['boxes'][:, :3], a_threshold=center_threshold),
                self.augment(true['boxes'][:, 3:-1], m_threshold=size_threshold),
                self.augment(true['boxes'][:, -1:], a_threshold=angle_threshold),
            )
            boxes_set = np.concatenate(augmented_boxes, axis=1)
            labels = np.where(
                np.random.random(n_boxes) < class_probability,
                np.random.choice(np.arange(n_classes), n_boxes),
                true['labels'],
            )
            scores = torch.ones(n_boxes)
            drop_mask = np.random.random(n_boxes) < drop_probability
            seen['boxes'].append(boxes_set[~drop_mask])
            seen['labels'].append(labels[~drop_mask])
            seen['scores'].append(scores[~drop_mask])

        return true, seen

    def generate_camera(
            self,
            angle_threshold: float,
            device: torch.device,
        ):
        point = np.random.uniform(0.0, 1.0, 3)
        forward_vector = np.array([0.5, 0.5, 0.5]) - point
        forward_vector = Rotation.from_rotvec(
            self.augment(np.zeros(3), a_threshold=angle_threshold),
        ).apply(forward_vector)
        rotation_matrix = self.create_rotation_matrix(forward_vector)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = point

        extrinsic = torch.tensor(
            np.linalg.inv(camera_pose),
            dtype=torch.float,
            device=device,
        )
        intrinsic_matrix = [
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        intrinsic = torch.tensor(intrinsic_matrix, device=device)
        camera = intrinsic @ extrinsic

        return camera

    def generate_camera_dataset(
            self,
            n: int,
            angle_threshold: float,
            device: torch.device,
        ):
        cameras = list()

        for _ in range(n):
            camera = self.generate_camera(
                angle_threshold=angle_threshold,
                device=device,
            )
            cameras.append(camera)

        return torch.stack(cameras, dim=0)

    def update_box_dataset_with_cameras(
            self,
            seen,
        ):
        for i in range(len(seen['boxes'])):
            if not len(seen['boxes'][i]):
                continue
            mask = Camera.check_boxes_in_camera_fov(
                boxes=seen['boxes'][i],
                camera=seen['cameras'][i],
            )
            for key in ['boxes', 'labels', 'scores']:
                seen[key][i] = seen[key][i][mask]

        for i, camera in enumerate(seen['cameras']):
            seen_boxes_set = seen['boxes'][i]
            seen_projections_set = Camera.project_boxes_onto_camera_plane(
                boxes=seen_boxes_set,
                camera=camera,
                mode='minmax',
            )
            seen['projections_set'].append(seen_projections_set)

    def setup(
            self,
            angle_threshold: float = 0.3,
            center_std: float = 0.2,
            center_threshold: float = 0.02,
            class_probability: float = 0.1,
            drop_probability: float = 0.2,
            size_mean: float = 0.05,
            size_std: float = 0.02,
            size_threshold: float = 0.1,
        ):
        self.true, self.seen = self.generate_box_dataset(
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

        cameras = self.generate_camera_dataset(
            n=self.n_cameras,
            angle_threshold=0.3,
            device=self.device,
        )

        self.seen['cameras'] = cameras
        self.update_box_dataset_with_cameras(seen=self.seen)

        initial_boxes = \
            torch.cat(tuple(filter(lambda x: len(x), self.seen['boxes'])))
        center_mean = initial_boxes[:, :3].mean(axis=0)
        center_std = initial_boxes[:, :3].std(axis=0)
        size_mean = initial_boxes[:, 3:-1].mean(axis=0)
        size_std = initial_boxes[:, 3:-1].std(axis=0)

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

        initial_scores = np.random.random((self.n_boxes, self.n_classes + 1))
        initial_scores[:, -1] = 0.0
        initial_scores = \
            torch.tensor(initial_scores, dtype=torch.float, device=self.device)
        optimized_scores = initial_scores.clone().detach()
        optimized_scores.requires_grad = True

        self.optimized = {
            'boxes': optimized_boxes,
            'scores': optimized_scores,
        }

    def get_constants(self):
        constants = {
            'n_boxes': self.n_boxes,
            'n_cameras': self.n_cameras,
            'n_classes': self.n_classes,
        }

        return constants
