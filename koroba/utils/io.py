from typing import Union

import numpy as np
import open3d as o3d
from torch import Tensor

from koroba.utils import Box


def write_bounding_box(
        filename: str,
        box: Union[np.ndarray, Tensor],
    ) -> None:
    assert len(box) == 7 or len(box) == 8
    if type(box) is Tensor:
        box = box.detach().cpu().numpy()

    if len(box) == 7:
        box = Box.seven2eight(box)

    box_point_cloud = o3d.geometry.PointCloud()
    box_point_cloud.points = o3d.utility.Vector3dVector(box)

    o3d.io.write_point_cloud(
        filename=filename,
        pointcloud=box_point_cloud,
    )


def read_bounding_box(
        filename: str,
    ) -> o3d.geometry.OrientedBoundingBox:
    cloud = o3d.io.read_point_cloud(filename)
    points = np.asarray(cloud.points)
    horizontal_bbox = Box.estimate_horizontal_bounding_box(points)

    return horizontal_bbox
