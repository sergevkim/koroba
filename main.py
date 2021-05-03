from argparse import ArgumentParser

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from koroba.losses import BoxMatchingLoss
from koroba.utils import (
    Camera,
    Randomizer,
    SyntheticData as SynData,
    Visualizer,
)


def get_device():
    assert torch.cuda.is_available()
    return torch.device('cuda:0')


def optimize_boxes(
        predicted,
        n_boxes: int,
        n_classes: int,
        n_steps: int = 200,
        no_object_weight: float = 0.4,
    ):
    device = get_device()

    initial_boxes = \
        np.concatenate(tuple(filter(lambda x: len(x), predicted['boxes'])))
    center_mean = np.mean(initial_boxes[:, :3], axis=0)
    center_std = np.std(initial_boxes[:, :3], axis=0)
    size_mean = np.mean(initial_boxes[:, 3: -1], axis=0)
    size_std = np.std(initial_boxes[:, 3: -1], axis=0)

    to_concat = (
        np.random.normal(center_mean, center_std, (n_boxes, 3)),
        np.abs(np.random.normal(size_mean, size_std, (n_boxes, 3))),
        np.random.uniform(.0, 2 * np.pi, (n_boxes, 1)),
    )
    initial_boxes = np.concatenate(to_concat, axis=1)
    initial_boxes[:, 3: -1] = np.log(initial_boxes[:, 3: -1])
    initial_boxes = \
        torch.tensor(initial_boxes, dtype=torch.float, device=device)
    boxes = initial_boxes.clone().detach()
    boxes.requires_grad = True

    initial_scores = np.random.random((n_boxes, n_classes + 1))
    initial_scores[:, -1] = .0
    initial_scores = \
        torch.tensor(initial_scores, dtype=torch.float, device=device)
    scores = initial_scores.clone().detach()
    scores.requires_grad = True

    for i in range(len(predicted['boxes'])):
        predicted['boxes'][i] = torch.tensor(
            predicted['boxes'][i],
            dtype=torch.float,
            device=device,
        )
        predicted['labels'][i] = torch.tensor(
            predicted['labels'][i],
            dtype=torch.long,
            device=device,
        )

    optimizer = torch.optim.Adam([boxes, scores], lr=0.01)

    for i in range(n_steps):
        i_loss = torch.tensor(.0, dtype=torch.float, device=device)

        for j in range(len(predicted['boxes'])):
            optimizer.zero_grad()
            match_boxes_loss, rows = BoxMatchingLoss.calculate_3d(
                p_boxes=predicted['boxes'][j],
                p_labels=predicted['labels'][j],
                boxes=boxes,
                scores=scores,
            )
            visible_index = Camera.check_boxes_in_camera_fov(
                boxes=boxes.detach().cpu().numpy(),
                camera=predicted['cameras'][j],
            )
            no_object_index = np.ones(n_boxes, dtype=np.bool)
            no_object_index[rows] = False
            no_object_index = np.logical_and(visible_index, no_object_index)
            n_no_object = np.sum(no_object_index)
            no_object_nll = F.cross_entropy(
                scores[no_object_index],
                torch.ones(
                    int(n_no_object),
                    dtype=torch.long,
                    device=scores.device,
                ) * n_classes,
                reduction='none',
            )
            n_matched = len(rows)
            loss = (
                torch.sum(match_boxes_loss) +
                torch.sum(no_object_nll) * no_object_weight
            )
            loss = loss / max(n_matched + n_no_object, 1)
            i_loss += loss

        i_loss = i_loss / len(predicted['boxes'])
        i_loss.backward()
        optimizer.step()
        print(f'i: {i};  loss: {i_loss.detach().cpu().numpy()}')

    boxes = boxes.detach().cpu().numpy()
    boxes[:, 3: -1] = np.exp(boxes[:, 3: -1])
    scores = torch.softmax(scores, dim=1).detach().cpu().numpy()
    return {
        'boxes': boxes,
        'labels': np.argmax(scores, axis=1),
        'scores': np.max(scores, axis=1)
    }


def run_box_experiment(
        n: int = 10,
        n_boxes: int = 4,
        n_classes: int = 10,
    ):
    true, predicted = SynData.generate_box_dataset(
        n=n,
        n_boxes=n_boxes,
        n_classes=n_classes,
        center_std=0.2,
        size_mean=0.05,
        size_std=0.02,
        class_probability=0.1,
        drop_probability=0.2,
        center_threshold=0.02,
        size_threshold=0.3,
        angle_threshold=0.3,
    )
    cameras = SynData.generate_camera_dataset(
        n=n,
        angle_threshold=.3,
    )
    predicted['cameras'] = cameras
    SynData.update_box_dataset_with_cameras(
        seen=predicted,
        proj=False,
    )
    print('predicted boxes:')

    for i in range(len(predicted['boxes'])):
        print('box set:')
        for j in range(len(predicted['boxes'][i])):
            string = (
                f"{predicted['boxes'][i][j]} |"
                f"{predicted['labels'][i][j]} |"
                f"{predicted['scores'][i][j]}"
            )
            print(string)

    # TODO: + 10 here is for complication of current experiments
    optimized = optimize_boxes(
        predicted,
        n_boxes=n_boxes + 10,
        n_classes=n_classes,
        no_object_weight=0.4,
    )
    print('true boxes:')
    for i in range(len(true['boxes'])):
        print(true['boxes'][i], '|', true['labels'][i])
    print('boxes:')
    for i in range(len(optimized['boxes'])):
        string = (
            f"{optimized['boxes'][i]} |"
            f"{optimized['labels'][i]} |"
            f"{optimized['scores'][i]}"
        )
        print(string)

    return optimized


def main(args):
    Randomizer.set_seed()
    np.set_printoptions(
        precision=5,
        suppress=True,
        sign=' ',
    )
    optimized = run_box_experiment()
    boxes = optimized['boxes']
    for i, box in enumerate(boxes):
        print(i, box.shape)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
