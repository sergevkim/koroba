from typing import Union

import numpy as np
import open3d as o3d
from torch import Tensor

from koroba.utils import Box


class Logger:
    @staticmethod
    def write_bounding_box(
            filename: str,
            box: Union[np.ndarray, Tensor],
        ) -> None:
        assert len(box) == 7 or len(box) == 8
        if len(box) == 7:
            box = Box.seven2eight(box)

        box_point_cloud = o3d.geometry.PointCloud()
        box_point_cloud.points = o3d.utility.Vector3dVector(box)

        o3d.io.write_point_cloud(
            filename=filename,
            pointcloud=box_point_cloud,
        )
