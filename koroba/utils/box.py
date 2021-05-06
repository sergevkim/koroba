import copy
from typing import Union

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor


class Box:
    '''
    Translates different version of 3d horizontal box definition:
        7 coords: 3 for center, 3 for lengths size, 1 for angle
        8 coords: 8 points
        2 coords: lower left front point and the opposite to it
    to each other.

    Also can estimate horizontal bounding box (z axis oriented only)
    on the all point cloud.
    '''
    @staticmethod
    def seven2eight(box: Union[np.ndarray, Tensor]):
        # TODO vectorize
        center = box[:3]
        sizes = box[3:6]
        alpha = box[6]
        rotation_matrix = [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1],
        ]
        rotation = Rotation.from_matrix(rotation_matrix)

        vertices = list()

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mask = ((-1) ** i, (-1) ** j, (-1) ** k)
                    if type(box) is Tensor:
                        mask = torch.tensor(mask)
                    print('!', center, sizes, mask)
                    vertex = center + rotation.apply(sizes * mask / 2)
                    vertices.append(vertex)

        if type(box) is Tensor:
            return torch.stack(vertices, dim=0)
        else:
            return np.array(vertices)

    @staticmethod
    def eight2seven(vertices: Union[np.ndarray, Tensor]):
        points = o3d.utility.Vector3dVector(np.array(vertices))
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(points)
        center = bbox.center
        extent = bbox.extent
        R = bbox.R

        import ipdb; ipdb.set_trace()

        '''
        center = vertices.sum(axis=0)
        sizes = vertices.sum(axis=0)
        alpha = np.array(np.pi / 4)[None, ...]

        if type(vertices) is Tensor:
            alpha = torch.tensor(alpha)
            box = torch.cat((center, sizes, alpha), dim=0)
        else:
            box = np.concatenate((center, sizes, alpha), axis=0)
        '''

        return box

    @staticmethod
    def estimate_horizontal_bounding_box(points: np.ndarray):
        n = len(points)
        points_lower = copy.deepcopy(points)
        points_upper = copy.deepcopy(points)

        z_max = min(points_lower[:, 2])
        z_min = max(points_upper[:, 2])
        points_lower[:, 2] = z_min
        points_upper[:, 2] = z_max

        points = np.concatenate((points_lower, points_upper))
        flat_points = o3d.utility.Vector3dVector(points)

        horizontal_bounding_box = \
            o3d.geometry.OrientedBoundingBox.create_from_points(flat_points)

        return horizontal_bounding_box


if __name__ == '__main__':
    box = np.array([0, 0, 0, 1, 2, 3, np.pi / 6])
    eight = Box.seven2eight(box)
    print(f'eight:\n{eight}\n')
    print(eight.shape)
