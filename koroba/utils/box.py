import copy

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from koroba.utils.constants import LINES


class Box:
    '''
    Translates different version of 3d horizontal box definition:
        7 coords: 3 for center, 3 for lengths size, 1 for angle
        8 coords: 8 points
        2 coords: lower left front point and the opposite to it
    to each other
    '''
    @staticmethod
    def seven2two(box):
        # TODO check coordinates
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

        two_points = (
            center + rotation.apply(-sizes / 2),
            center + rotation.apply(sizes / 2),
        )

        return np.array(two_points)

    @classmethod
    def seven2eight(cls, box):
        two_points = cls.seven2two(box)
        eight_points = list()

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    x = two_points[i][0]
                    y = two_points[j][1]
                    z = two_points[k][2]
                    eight_points.append((x, y, z))

        return np.array(eight_points)

    @staticmethod
    def eight2seven(box):
        pass

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

    @staticmethod
    def get_geometry(
            bbox: o3d.geometry.OrientedBoundingBox,
            spheres_flag: bool = False,
        ):
        bbox_vertices = np.asarray(bbox.get_box_points())
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_vertices),
            lines=o3d.utility.Vector2iVector(LINES),
        )

        if spheres_flag:
            bbox_spheres = add_points([], bbox_vertices, color='r', size='big')
            bbox_geometry = [line_set] + bbox_spheres
        else:
            bbox_geometry = [line_set]

        return bbox_geometry


if __name__ == '__main__':
    box = np.array([0, 0, 0, 1, 2, 3, np.pi / 4])
    two = Box.seven2two(box)
    print(f'two:\n{two}\n')
    eight = Box.seven2eight(box)
    print(f'eight:\n{eight}\n')
