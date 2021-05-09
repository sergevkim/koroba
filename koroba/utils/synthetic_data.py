from collections import defaultdict

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from koroba.utils import Camera


def augment(
        x,
        m_threshold=None,
        a_threshold=None,
    ):
    if m_threshold is not None:
        x = x * np.random.uniform(1. - m_threshold, 1. + m_threshold, x.shape)
    if a_threshold is not None:
        x = x + np.random.uniform(-a_threshold, a_threshold, x.shape)

    return x


def create_rotation_matrix(forward_vector):
    v1 = forward_vector / np.linalg.norm(forward_vector)
    v2 = np.cross([0.0, 0.0, 1.0], v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.cross(v1, v2)

    #three axes; v1 is collinear to the forward vect
    return np.stack((v2, v3, v1), axis=1)


class SyntheticData:
    @staticmethod
    def generate_layout_dataset(
            n,
            center_threshold,
            size_threshold,
            angle_threshold,
        ):
        layout = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0])
        layouts = np.stack([layout] * n)
        augmented_layouts = (
            augment(layouts[:, :3], a_threshold=center_threshold),
            augment(layouts[:, 3: -1], m_threshold=size_threshold),
            augment(layouts[:, -1:], a_threshold=angle_threshold),
        )
        layouts = np.concatenate(augmented_layouts, axis=1)

        return layout, layouts

    @staticmethod
    def generate_camera(
            angle_threshold: float,
            device: torch.device,
        ):
        point = np.random.uniform(0.0, 1.0, 3)
        forward_vector = np.array([0.5, 0.5, 0.5]) - point
        forward_vector = Rotation.from_rotvec(
            augment(np.zeros(3), a_threshold=angle_threshold),
        ).apply(forward_vector)
        rotation_matrix = create_rotation_matrix(forward_vector)
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

    @classmethod
    def generate_camera_dataset(
            cls,
            n: int,
            angle_threshold: float,
            device: torch.device,
        ):
        cameras = list()

        for _ in range(n):
            camera = cls.generate_camera(
                angle_threshold=angle_threshold,
                device=device,
            )
            cameras.append(camera)

        return torch.stack(cameras, dim=0)

    @staticmethod
    def generate_box_dataset(
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
                augment(true['boxes'][:, :3], a_threshold=center_threshold),
                augment(true['boxes'][:, 3:-1], m_threshold=size_threshold),
                augment(true['boxes'][:, -1:], a_threshold=angle_threshold),
            )
            boxes_set = np.concatenate(augmented_boxes, axis=1)
            labels = np.where(
                np.random.random(n_boxes) < class_probability,
                np.random.choice(np.arange(n_classes), n_boxes),
                true['labels'],
            )
            scores = np.ones(n_boxes)
            drop_mask = np.random.random(n_boxes) < drop_probability
            seen['boxes'].append(boxes_set[~drop_mask])
            seen['labels'].append(labels[~drop_mask])
            seen['scores'].append(scores[~drop_mask])

        return true, seen

    @staticmethod
    def update_box_dataset_with_cameras(
            seen,
            proj: bool = False,
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

        if proj:
            seen['projections'] = list()

            for i, camera in enumerate(seen['cameras']):
                boxes_set = seen['boxes'][i]
                proj = Camera.project_boxes_onto_camera_plane(
                    camera=camera,
                    boxes_set=boxes_set,
                )
                seen['projections'].append(proj)

