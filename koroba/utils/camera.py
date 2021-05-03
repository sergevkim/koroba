import numpy as np
from torch import Tensor

from koroba.utils import Box


class Camera:
    @staticmethod
    def check_boxes_in_camera_fov(
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
        assert box.shape == (7, 3)
        vertices = Box.seven2eight(box).T
        assert vertices.shape == (8, 3)
        to_concat = (
            vertices,
            np.ones(shape=(1, len(vertices_3d))),
        )
        vertices = np.concatenate(to_concat, axis=0)
        vertices_projection = camera @ vertices

        return vertices_projection

    @classmethod
    def project_boxes_onto_camera_plane(
            cls,
            boxes: Tensor,
            camera: np.ndarray,
        ) -> Tensor:
        boxes_vertices_list = list()

        for box in boxes:
            vertices_projection = cls.project_single_box_onto_camera_plane(
                box=box,
                camera=camera,
            )
            point_min = vertices_projection.min(axis=0)
            point_max = vertices_projection.max(axis=0)
            box_projection = np.array((point_min, point_max))
            boxes_projections.append(box_projection)

        return boxes_projections
