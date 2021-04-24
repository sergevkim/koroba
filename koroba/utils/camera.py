class Camera:
    @staticmethod
    def check_boxes_in_camera_fov(boxes, camera):
        center_3d = boxes[:, :3].T
        to_concat = (
            center_3d,
            np.ones((1, len(boxes))),
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