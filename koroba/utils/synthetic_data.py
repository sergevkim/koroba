from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation


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
    v2 = np.cross([.0, .0, 1.], v1)
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
        layout = np.array([.5, .5, .5, 1., 1., 1., .0])
        layouts = np.stack([layout] * n)
        augmented_layouts = (
            augment(layouts[:, :3], a_threshold=center_threshold),
            augment(layouts[:, 3: -1], m_threshold=size_threshold),
            augment(layouts[:, -1:], a_threshold=angle_threshold),
        )
        layouts = np.concatenate(augmented_layouts, axis=1)

        return layout, layouts

    @staticmethod
    def generate_camera_dataset(
            n,
            angle_threshold,
        ):
        cameras = []
        for _ in range(n):
            point = np.random.uniform(.0, 1., 3)
            forward_vector = np.array([.5, .5, .5]) - point
            forward_vector = Rotation.from_rotvec(
                augment(np.zeros(3), a_threshold=angle_threshold),
            ).apply(forward_vector)
            rotation_matrix = create_rotation_matrix(forward_vector)
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = rotation_matrix
            camera_pose[:3, 3] = point

            extrinsic = np.linalg.inv(camera_pose)
            intrinsic = np.array([
                [.5, .0, .5, .0],
                [.0, .5, .5, .0],
                [.0, .0, 1., .0],
            ])
            camera = intrinsic @ extrinsic
            cameras.append(camera)

        return cameras

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
            np.random.normal(.5, center_std, (n_boxes, 3)),
            np.abs(np.random.normal(size_mean, size_std, (n_boxes, 3))),
            np.random.uniform(.0, 2 * np.pi, (n_boxes, 1))
        )
        boxes = np.concatenate(to_concat, axis=1)
        true = {
            'boxes': boxes,
            'labels': np.random.choice(np.arange(n_classes), n_boxes)
        }

        predicted = defaultdict(list)

        for _ in range(n):
            augmented_boxes = (
                augment(true['boxes'][:, :3], a_threshold=center_threshold),
                augment(true['boxes'][:, 3: -1], m_threshold=size_threshold),
                augment(true['boxes'][:, -1:], a_threshold=angle_threshold),
            )
            boxes = np.concatenate(augmented_boxes, axis=1)
            labels = np.where(
                np.random.random(n_boxes) < class_probability,
                np.random.choice(np.arange(n_classes), n_boxes),
                true['labels']
            )
            scores = np.ones(n_boxes)
            drop_mask = np.random.random(n_boxes) < drop_probability
            predicted['boxes'].append(boxes[~drop_mask])
            predicted['labels'].append(labels[~drop_mask])
            predicted['scores'].append(scores[~drop_mask])

        return true, predicted
