import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from koroba.utils import Box


class Camera:
    @staticmethod
    def check_boxes_in_camera_fov( #TODO projection on 8 points
            boxes: Tensor,
            camera: np.ndarray,
        ):
        center_3d = boxes[:, :3].T
        to_concat = (
            center_3d,
            np.ones(shape=(1, len(boxes))),
        )
        center_3d = np.concatenate(to_concat, axis=0)
        x, y, z = camera @ center_3d
        x /= z
        y /= z
        check = np.logical_and.reduce((
            z >= .0,
            x >= .0,
            x <= 1.,
            y >= .0,
            y <= 1.
        ))

        return check

    @staticmethod
    def project_single_box_onto_camera_plane(
            box: Tensor,
            camera: np.ndarray,
        ) -> Tensor:
        assert box.shape == (7, )
        vertices = Box.seven2eight(box)
        assert vertices.shape == (8, 3)
        to_concat = (
            vertices.T,
            np.ones(shape=(1, len(vertices))),
        )
        vertices = np.concatenate(to_concat, axis=0)
        x, y, z = camera @ vertices
        x = torch.tensor(x / z)
        y = torch.tensor(y / z)

        return x, y

    @classmethod #TODO vectorize
    def project_boxes_onto_camera_plane(
            cls,
            boxes: Tensor,
            camera: np.ndarray,
            mode: str = 'minmax',
        ) -> Tensor:
        boxes_projections = list()

        for box in boxes:
            x, y = cls.project_single_box_onto_camera_plane(
                box=box,
                camera=camera,
            )
            if mode == 'minmax':
                x_min, _ = x.min(axis=0)
                y_min, _ = y.min(axis=0)
                x_max, _ = x.max(axis=0)
                y_max, _ = y.max(axis=0)
            elif mode == 'attention':
                x_min = x @ F.softmin(x, dim=0)
                y_min = y @ F.softmin(y, dim=0)
                x_max = x @ F.softmax(x, dim=0)
                y_max = y @ F.softmax(y, dim=0)

            point_min = np.array((x_min, y_min))
            point_max = np.array((x_max, y_max))
            box_projection = np.array((point_min, point_max))
            boxes_projections.append(box_projection)

        return torch.tensor(boxes_projections)
