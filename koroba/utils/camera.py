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

    #TODO
    @staticmethod
    def project_boxes_onto_camera_plane(
            boxes_set: Tensor,
            camera: np.ndarray,
        ) -> Tensor:
        boxes_vertices_list = list()

        for box in boxes_set:
            box_vertices = Box.seven2eight(box)
            boxes_vertices_list.append(box_vertices)

        for v in boxes_vertices:
            v = v.T
            to_concat = (
                v,
                np.ones(shape=(1, len(boxes))),
            )
            center_3d = np.concatenate(to_concat, axis=0)