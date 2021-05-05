import numpy as np

from koroba.utils import Camera, SyntheticData as SynData, Randomizer


Randomizer.set_seed()


def test_projection():
    camera = SynData.generate_camera(angle_threshold=0.3)
    assert camera.shape == (3, 4)

    to_concat = (
        np.random.normal(0.5, 0.2, (10, 3)),
        np.abs(np.random.normal(0.05, 0.02, (10, 3))),
        np.random.uniform(0.0, 2 * np.pi, (10, 1))
    )
    boxes = np.concatenate(to_concat, axis=1)

    print(boxes[0])

    result = Camera.project_single_box_onto_camera_plane(
        box=boxes[0],
        camera=camera,
    )
    print(result.shape)


if __name__ == '__main__':
    test_projection()
