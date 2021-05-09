import numpy as np
import torch
from torch import Tensor

from koroba.utils import (
    Box,
    Randomizer,
)


Randomizer.set_seed()


def test_box3d_format_translation_shapes():
    box = torch.tensor([0, 0, 0, 1, 2, 3, np.pi / 4])
    vertices = Box.box3d_to_vertices3d(box)
    assert vertices.shape == (8, 3)
    box_tilde = Box.vertices3d_to_box3d(vertices)
    assert box_tilde.shape == (7, )


def test_box3d_format_translation():
    box = torch.tensor([0, 0, 0, 1, 2, 3, np.pi / 4])
    vertices = Box.box3d_to_vertices3d(box)
    assert vertices.shape == (8, 3)
    box_tilde = Box.vertices3d_to_box3d(vertices)
    assert box_tilde.shape == (7, )
    diff = np.linalg.norm(box_tilde - box)
    assert diff < 1e-6


def test_box2d_format_translation_shapes():
    vertices = torch.tensor([
        [1.0, 1.0],
        [2.0, 3.0],
    ])
    box_tilde = Box.vertices2d_to_box2d(vertices)
    print(box_tilde)
    assert box_tilde.shape == (5, )


if __name__ == '__main__':
    test_box3d_format_translation_shapes()
    test_box3d_format_translation()
    test_box2d_format_translation_shapes()

