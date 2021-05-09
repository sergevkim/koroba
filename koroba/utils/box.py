import copy
from typing import Union

import numpy as np
import open3d as o3d
import torch
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
    def box3d_to_vertices3d(box: Tensor):
        # TODO vectorize
        center = box[:3]
        sizes = box[3:6]
        alpha = box[6]

        rotation_matrix = torch.tensor([
            [torch.cos(alpha), -torch.sin(alpha), 0],
            [torch.sin(alpha), torch.cos(alpha), 0],
            [0, 0, 1],
        ]).to(box.device)

        vertices = list()

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mask = ((-1) ** i, (-1) ** j, (-1) ** k)
                    if type(box) is Tensor:
                        mask = torch.tensor(mask).to(box.device)
                    vertex = center + rotation_matrix @ (sizes * mask / 2)
                    vertices.append(vertex)

        return torch.stack(vertices, dim=0)

    @staticmethod
    def vertices3d_to_box3d(vertices: Tensor):
        center = vertices.sum(axis=0)
        # right upper left lower in xy plane
        right_index, far_index, high_index = vertices.argmax(axis=0)
        left_index, _, low_index = vertices.argmin(axis=0)
        x_right, y_right, _ = vertices[right_index]
        x_far, y_far, _ = vertices[far_index]
        x_left, y_left, _ = vertices[left_index]
        _, _, z_high = vertices[high_index]
        _, _, z_low = vertices[low_index]

        far_right_x_delta = np.abs(x_right - x_far)
        far_right_y_delta = np.abs(y_right - y_far)
        far_left_x_delta = np.abs(x_left - x_far)
        far_left_y_delta = np.abs(y_left - y_far)
        angle = np.arctan(far_right_x_delta / far_right_y_delta)
        y_size = np.sqrt(far_right_x_delta ** 2 + far_right_y_delta ** 2)
        x_size = np.sqrt(far_left_x_delta ** 2 + far_left_y_delta ** 2)
        z_size = z_high - z_low

        extent = torch.tensor([x_size, y_size, z_size])
        angle = torch.tensor([angle])
        box = torch.cat((center, extent, angle), dim=0)

        return box

    @staticmethod
    def vertices2d_to_box2d(vertices: Tensor):
        angle = torch.tensor([0])
        center = vertices.mean(axis=0)
        extent = torch.abs(vertices[1] - vertices[0]) / 2
        box = torch.cat((center, extent, angle), dim=0)

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
    eight = Box.box3d_to_vertices3d(box)
    print(f'eight:\n{eight}\n')
    print(eight.shape)
