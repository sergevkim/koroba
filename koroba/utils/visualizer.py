from typing import Tuple, Union

import numpy as np
import open3d as o3d
import plotly.graph_objects as go

from koroba.utils.constants import LINES


class Visualizer:
    @staticmethod
    def get_geometry(
            bbox: o3d.geometry.OrientedBoundingBox,
            color: Union[str, Tuple[float]] = (0.0, 0.0, 1.0),
            spheres_flag: bool = False,
        ):
        bbox_vertices = np.asarray(bbox.get_box_points())
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_vertices),
            lines=o3d.utility.Vector2iVector(LINES),
        )

        if color == 'red':
            color = (1.0, 0.0, 0.0)
        elif color == 'green':
            color = (0.0, 1.0, 0.0)
        elif color == 'blue':
            color = (0.0, 0.0, 1.0)

        line_set.paint_uniform_color(np.array(color))

        if spheres_flag:
            bbox_spheres = add_points([], bbox_vertices, color='r', size='big')
            bbox_geometry = [line_set] + bbox_spheres
        else:
            bbox_geometry = [line_set]

        return bbox_geometry

    @staticmethod
    def draw_geometries(geometries):
        graph_objects = list()

        for geometry in geometries:
            geometry_type = geometry.get_geometry_type()

            if geometry_type == o3d.geometry.Geometry.Type.PointCloud:
                points = np.asarray(geometry.points)
                colors = None
                if geometry.has_colors():
                    colors = np.asarray(geometry.colors)
                elif geometry.has_normals():
                    normals = np.asarray(geometry.normals)
                    colors = (0.5, 0.5, 0.5) + normals * 0.5
                else:
                    geometry.paint_uniform_color((1.0, 0.0, 0.0))
                    colors = np.asarray(geometry.colors)

                scatter_3d = go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=colors,
                    ),
                )
                graph_objects.append(scatter_3d)

            if geometry_type == o3d.geometry.Geometry.Type.TriangleMesh:
                triangles = np.asarray(geometry.triangles)
                vertices = np.asarray(geometry.vertices)
                colors = None

                if geometry.has_triangle_normals():
                    triangle_normals = np.asarray(geometry.triangle_normals)
                    colors = (0.5, 0.5, 0.5) + triangle_normals * 0.5
                    colors = tuple(map(tuple, colors))
                else:
                    colors = (1.0, 0.0, 0.0)

                mesh_3d = go.Mesh3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    i=triangles[:, 0],
                    j=triangles[:, 1],
                    k=triangles[:, 2],
                    facecolor=colors,
                    opacity=0.50,
                )
                graph_objects.append(mesh_3d)

            if geometry_type == o3d.geometry.Geometry.Type.LineSet:
                points = np.asarray(geometry.points)
                lines = np.asarray(geometry.lines)

                if geometry.has_colors():
                    colors = np.asarray(geometry.colors)
                else:
                    colors = 'darkblue'

                x0 = points[:, 0][lines[:, 0]]
                y0 = points[:, 1][lines[:, 0]]
                z0 = points[:, 2][lines[:, 0]]
                x1 = points[:, 0][lines[:, 1]]
                y1 = points[:, 1][lines[:, 1]]
                z1 = points[:, 2][lines[:, 1]]

                for i in range(len(lines)):
                    bounding_box = go.Scatter3d(
                        x=[x0[i], x1[i]],
                        y=[y0[i], y1[i]],
                        z=[z0[i], z1[i]],
                        marker=dict(
                            size=4,
                            color=z0,
                            colorscale='Viridis',
                        ),
                        line=dict(
                            color=colors,
                            width=2
                        )
                    )
                    graph_objects.append(bounding_box)

        fig = go.Figure(
            data=graph_objects,
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=True),
                    yaxis=dict(visible=True),
                    zaxis=dict(visible=True)
                )
            )
        )
        fig.show()
