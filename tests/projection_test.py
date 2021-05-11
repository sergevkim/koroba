import numpy as np
import torch
from torch import Tensor

from koroba.datamodules import SyntheticDataModule
from koroba.utils import (
    Box,
    Camera,
    Randomizer,
)


Randomizer.set_seed()


def test_box_in_camera_fov_checking():
    camera = SyntheticDataModule.generate_camera(angle_threshold=0.3, device=torch.device('cpu'))
    assert camera.shape == (3, 4)

    to_concat = (
        torch.tensor(np.random.normal(0.5, 0.2, (10, 3)), dtype=torch.float),
        torch.tensor(np.abs(np.random.normal(0.05, 0.02, (10, 3))), dtype=torch.float),
        torch.tensor(np.random.uniform(0.0, 2 * np.pi, (10, 1)), dtype=torch.float),
    )
    boxes = torch.cat(to_concat, axis=1)

    Camera.check_boxes_in_camera_fov(
        boxes=boxes,
        camera=camera,
    )


def test_single_box_projection():
    camera = SyntheticDataModule.generate_camera(angle_threshold=0.3, device=torch.device('cpu'))
    assert camera.shape == (3, 4)

    to_concat = (
        torch.tensor(np.random.normal(0.5, 0.2, (10, 3)), dtype=torch.float),
        torch.tensor(np.abs(np.random.normal(0.05, 0.02, (10, 3))), dtype=torch.float),
        torch.tensor(np.random.uniform(0.0, 2 * np.pi, (10, 1)), dtype=torch.float),
    )
    boxes = torch.cat(to_concat, axis=1)

    x, y = Camera.project_single_box_onto_camera_plane(
        box=boxes[0],
        camera=camera,
    )
    assert x.shape == (8, )
    assert y.shape == (8, )
    assert type(x) is Tensor
    assert type(y) is Tensor


def test_boxes_projection():
    camera = SyntheticDataModule.generate_camera(angle_threshold=0.3, device=torch.device('cpu'))
    assert camera.shape == (3, 4)

    to_concat = (
        torch.tensor(np.random.normal(0.5, 0.2, (10, 3)), dtype=torch.float),
        torch.tensor(np.abs(np.random.normal(0.05, 0.02, (10, 3))), dtype=torch.float),
        torch.tensor(np.random.uniform(0.0, 2 * np.pi, (10, 1)), dtype=torch.float),
    )
    boxes = torch.cat(to_concat, axis=1)

    boxes_projections = Camera.project_boxes_onto_camera_plane(
        boxes=boxes,
        camera=camera,
        mode='minmax',
    )
    assert boxes_projections.shape == (10, 5)
    assert type(boxes_projections) is Tensor

    boxes_projections = Camera.project_boxes_onto_camera_plane(
        boxes=boxes,
        camera=camera,
        mode='attention',
    )
    assert boxes_projections.shape == (10, 5)
    assert type(boxes_projections) is Tensor


if __name__ == '__main__':
    test_box_in_camera_fov_checking()
    test_single_box_projection()
    test_boxes_projection()
