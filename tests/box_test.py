import numpy as np
import torch
from torch import Tensor

from koroba.utils import (
    Box,
    Randomizer,
)


Randomizer.set_seed()


def test_box_format_translation_numpy_shapes():
    box = np.array([0, 0, 0, 1, 2, 3, np.pi / 4])
    eight = Box.seven2eight(box)
    assert eight.shape == (8, 3)
    seven = Box.eight2seven(eight)
    assert seven.shape == (7, )


def test_box_format_translation_torch_shapes():
    box = torch.tensor([0, 0, 0, 1, 2, 3, np.pi / 4])
    eight = Box.seven2eight(box)
    assert eight.shape == (8, 3)
    seven = Box.eight2seven(eight)
    assert seven.shape == (7, )


def test_box_format_translation_numpy():
    box = np.array([0, 0, 0, 1, 2, 3, np.pi / 4])
    eight = Box.seven2eight(box)
    assert eight.shape == (8, 3)
    seven = Box.eight2seven(eight)
    assert seven.shape == (7, )
    assert (seven == box).all()


def test_box_format_translation_torch():
    box = torch.tensor([0, 0, 0, 1, 2, 3, np.pi / 4])
    eight = Box.seven2eight(box)
    assert eight.shape == (8, 3)
    seven = Box.eight2seven(eight)
    assert seven.shape == (7, )
    assert (seven == box).all()


if __name__ == '__main__':
    test_box_format_translation_numpy_shapes()
    test_box_format_translation_torch_shapes()
    test_box_format_translation_numpy()
    test_box_format_translation_torch()
