import numpy as np
import torch
from torch import Tensor

from koroba.utils import (
    Box,
    Camera,
    SyntheticData as SynData,
    Randomizer,
)


Randomizer.set_seed()


def test_box_in_camera_fov_checking():
    camera = SynData.generate_camera(angle_threshold=0.3)
    assert camera.shape == (3, 4)

    to_concat = (
        np.random.normal(0.5, 0.2, (10, 3)),
        np.abs(np.random.normal(0.05, 0.02, (10, 3))),
        np.random.uniform(0.0, 2 * np.pi, (10, 1))
    )
    boxes = np.concatenate(to_concat, axis=1)

    Camera.check_boxes_in_camera_fov(
        boxes=boxes,
        camera=camera,
    )


def test_single_box_projection():
    camera = SynData.generate_camera(angle_threshold=0.3)
    assert camera.shape == (3, 4)

    to_concat = (
        np.random.normal(0.5, 0.2, (10, 3)),
        np.abs(np.random.normal(0.05, 0.02, (10, 3))),
        np.random.uniform(0.0, 2 * np.pi, (10, 1))
    )
    boxes = np.concatenate(to_concat, axis=1)
    boxes = torch.tensor(boxes, dtype=torch.float)

    x, y = Camera.project_single_box_onto_camera_plane(
        box=boxes[0],
        camera=camera,
    )
    assert x.shape == (8, )
    assert y.shape == (8, )
    assert type(x) is Tensor
    assert type(y) is Tensor


def test_boxes_projection():
    camera = SynData.generate_camera(angle_threshold=0.3)
    assert camera.shape == (3, 4)

    to_concat = (
        np.random.normal(0.5, 0.2, (10, 3)),
        np.abs(np.random.normal(0.05, 0.02, (10, 3))),
        np.random.uniform(0.0, 2 * np.pi, (10, 1))
    )
    boxes = np.concatenate(to_concat, axis=1)
    boxes = torch.tensor(boxes, dtype=torch.float)

    boxes_projections = Camera.project_boxes_onto_camera_plane(
        boxes=boxes,
        camera=camera,
        mode='minmax',
    )
    assert boxes_projections.shape == (10, 2, 2)
    assert type(boxes_projections) is Tensor

    boxes_projections = Camera.project_boxes_onto_camera_plane(
        boxes=boxes,
        camera=camera,
        mode='attention',
    )
    assert boxes_projections.shape == (10, 2, 2)
    assert type(boxes_projections) is Tensor


if __name__ == '__main__':
    test_box_in_camera_fov_checking()
    test_single_box_projection()
    test_boxes_projection()
