import numpy as np
import open3d as o3d
import plotly.graph_objects as go


class Visualizer:
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
                            color='darkblue',
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
