import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import koroba.utils.iou as iou
from koroba.utils import (
    Camera,
    Randomizer,
    SyntheticData as SynData,
    Visualizer,
)


def get_device():
    assert torch.cuda.is_available()
    return torch.device('cuda:0')


def match_boxes(
        p_boxes,
        p_labels,
        boxes,
        scores,
    ):
    n_p_boxes = len(p_boxes)
    if not n_p_boxes:
        return torch.tensor(0.0, dtype=torch.float, device=boxes.device), []

    n_boxes = boxes.shape[0]
    to_concat = [
        boxes[:, :3],
        torch.exp(boxes[:, 3: -1]),
        boxes[:, -1:],
    ]
    exp_boxes = torch.cat(to_concat, dim=1)
    repeated_boxes = exp_boxes.repeat_interleave(n_p_boxes, 0)
    repeated_scores = scores.repeat_interleave(n_p_boxes, 0)
    repeated_p_boxes = p_boxes.repeat(n_boxes, 1)
    repeated_p_labels = p_labels.repeat(n_boxes)

    pairwise_giou = iou.calculate_3d_giou(
        box3d1=repeated_boxes[None, ...],
        box3d2=repeated_p_boxes[None, ...],
    )[0]
    pairwise_giou = pairwise_giou.reshape(n_boxes, n_p_boxes)
    # pairwise_l1 = torch.mean(torch.abs(repeated_boxes[:, :3] - repeated_p_boxes[:, :3]), dim=1)
    # pairwise_l1 = pairwise_l1.reshape(n_boxes, n_p_boxes)
    pairwise_nll = F.cross_entropy(
        repeated_scores,
        repeated_p_labels,
        reduction='none',
    )
    pairwise_nll = pairwise_nll.reshape(n_boxes, n_p_boxes)
    cost = pairwise_giou + pairwise_nll
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

    return cost[rows, columns], rows


def optimize_boxes(
        predicted,
        n_boxes,
        n_classes,
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

    for i in range(500):
        i_loss = torch.tensor(.0, dtype=torch.float, device=device)

        for j in range(len(predicted['boxes'])):
            optimizer.zero_grad()
            match_boxes_loss, rows = match_boxes(
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
    SynData.update_box_dataset_with_cameras(predicted)
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
            f"{optimized['boxes'][i][j]} |"
            f"{optimized['labels'][i][j]} |"
            f"{optimized['scores'][i][j]}"
        )
        print(string)


if __name__ == '__main__':
    Randomizer.set_seed()
    np.set_printoptions(precision=5, suppress=True, sign=' ')
    run_box_experiment()
