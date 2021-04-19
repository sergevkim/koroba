import torch
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from rotated_iou import calculate_3d_giou


def get_device():
    assert torch.cuda.is_available()
    return torch.device('cuda:0')


def augment(x, m_threshold=None, a_threshold=None):
    if m_threshold is not None:
        x = x * np.random.uniform(1. - m_threshold, 1. + m_threshold, x.shape)
    if a_threshold is not None:
        x = x + np.random.uniform(-a_threshold, a_threshold, x.shape)
    return x


def generate_layout_dataset(n, center_threshold, size_threshold, angle_threshold):
    layout = np.array([.5, .5, .5, 1., 1., 1., .0])
    layouts = np.stack([layout] * n)
    layouts = np.concatenate((
        augment(layouts[:, :3], a_threshold=center_threshold),
        augment(layouts[:, 3: -1], m_threshold=size_threshold),
        augment(layouts[:, -1:], a_threshold=angle_threshold)
    ), axis=1)
    return layout, layouts


def generate_box_dataset(
    n, n_boxes, n_classes, center_std, size_mean, size_std, class_probability, drop_probability,
    center_threshold, size_threshold, angle_threshold
):
    true = {
        'boxes': np.concatenate((
            np.random.normal(.5, center_std, (n_boxes, 3)),
            np.abs(np.random.normal(size_mean, size_std, (n_boxes, 3))),
            np.random.uniform(.0, 2 * np.pi, (n_boxes, 1))
        ), axis=1),
        'labels': np.random.choice(np.arange(n_classes), n_boxes)
    }

    predicted = defaultdict(list)
    for _ in range(n):
        boxes = np.concatenate((
            augment(true['boxes'][:, :3], a_threshold=center_threshold),
            augment(true['boxes'][:, 3: -1], m_threshold=size_threshold),
            augment(true['boxes'][:, -1:], a_threshold=angle_threshold)
        ), axis=1)
        labels = np.where(
            np.random.random(n_boxes) < class_probability,
            np.random.choice(np.arange(n_classes), n_boxes),
            true['labels']
        )
        scores = np.ones(n_boxes)
        drop_mask = np.random.random(n_boxes) < drop_probability
        predicted['boxes'].append(boxes[~drop_mask])
        predicted['labels'].append(labels[~drop_mask])
        predicted['scores'].append(scores[~drop_mask])
    return true, predicted


def create_rotation_matrix(forward_vector):
    v1 = forward_vector / np.linalg.norm(forward_vector)
    v2 = np.cross([.0, .0, 1.], v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = np.cross(v1, v2)
    return np.stack((v2, v3, v1), axis=1)


def check_boxes_in_camera(boxes, camera):
    center_3d = boxes[:, :3].T
    center_3d = np.concatenate((center_3d, np.ones((1, len(boxes)))), axis=0)
    x, y, z = np.matmul(camera, center_3d)
    x /= z
    y /= z
    return np.logical_and.reduce((
        z >= .0,
        x >= .0,
        x <= 1.,
        y >= .0,
        y <= 1.
    ))


def generate_camera_dataset(n, angle_threshold):
    cameras = []
    for _ in range(n):
        intrinsic = np.array([
            [.5, .0, .5, .0],
            [.0, .5, .5, .0],
            [.0, .0, 1., .0]
        ])
        point = np.random.uniform(.0, 1., 3)
        forward_vector = np.array([.5, .5, .5]) - point
        forward_vector = Rotation.from_rotvec(augment(np.zeros(3), a_threshold=angle_threshold)).apply(forward_vector)
        rotation_matrix = create_rotation_matrix(forward_vector)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = rotation_matrix
        camera_pose[:3, 3] = point
        extrinsic = np.linalg.inv(camera_pose)
        camera = np.matmul(intrinsic, extrinsic)
        cameras.append(camera)
    return cameras


def update_box_dataset_with_cameras(predicted):
    for i in range(len(predicted['boxes'])):
        if not len(predicted['boxes'][i]):
            continue
        mask = check_boxes_in_camera(predicted['boxes'][i], predicted['cameras'][i])
        for key in ['boxes', 'labels', 'scores']:
            predicted[key][i] = predicted[key][i][mask]


def match_boxes(p_boxes, p_labels, boxes, scores):
    n_p_boxes = p_boxes.shape[0]
    if not n_p_boxes:
        return torch.tensor(.0, dtype=torch.float, device=boxes.device), []

    n_boxes = boxes.shape[0]
    exp_boxes = torch.cat([boxes[:, :3], torch.exp(boxes[:, 3: -1]), boxes[:, -1:]], dim=1)
    repeated_boxes = exp_boxes.repeat_interleave(n_p_boxes, 0)
    repeated_scores = scores.repeat_interleave(n_p_boxes, 0)
    repeated_p_boxes = p_boxes.repeat(n_boxes, 1)
    repeated_p_labels = p_labels.repeat(n_boxes)

    pairwise_giou = calculate_3d_giou(repeated_boxes[None, ...], repeated_p_boxes[None, ...])[0]
    pairwise_giou = pairwise_giou.reshape(n_boxes, n_p_boxes)
    # pairwise_l1 = torch.mean(torch.abs(repeated_boxes[:, :3] - repeated_p_boxes[:, :3]), dim=1)
    # pairwise_l1 = pairwise_l1.reshape(n_boxes, n_p_boxes)
    pairwise_nll = torch.nn.functional.cross_entropy(repeated_scores, repeated_p_labels, reduction='none')
    pairwise_nll = pairwise_nll.reshape(n_boxes, n_p_boxes)
    cost = pairwise_giou + pairwise_nll
    rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

    return cost[rows, columns], rows


def optimize_layout(layouts):
    device = get_device()

    initial_layout = np.mean(layouts, axis=0)
    initial_layout[3: -1] = np.log(initial_layout[3: -1])
    initial_layout = torch.tensor(initial_layout, dtype=torch.float, device=device)
    layouts = torch.tensor(layouts, dtype=torch.float, device=device)
    layout = initial_layout.clone().detach()
    layout.requires_grad = True

    optimizer = torch.optim.Adam([layout], lr=.1)
    for i in range(100):
        optimizer.zero_grad()
        exp_layout = torch.cat([layout[:3], torch.exp(layout[3: -1]), layout[-1:]])
        loss = torch.mean(
            calculate_3d_giou(
                torch.stack([exp_layout] * layouts.shape[0])[None, ...],
                layouts[None, ...]
            )[0]
        )
        loss.backward()
        optimizer.step()
        print('i:', i, 'giou_loss:', loss.detach().cpu().numpy())
    layout = layout.detach().cpu().numpy()
    layout[3: -1] = np.exp(layout[3: -1])
    return layout


def optimize_boxes(predicted, n_boxes, n_classes, no_object_weight):
    device = get_device()

    initial_boxes = np.concatenate(tuple(filter(lambda x: len(x), predicted['boxes'])))
    center_mean = np.mean(initial_boxes[:, :3], axis=0)
    center_std = np.std(initial_boxes[:, :3], axis=0)
    size_mean = np.mean(initial_boxes[:, 3: -1], axis=0)
    size_std = np.std(initial_boxes[:, 3: -1], axis=0)

    initial_boxes = np.concatenate((
        np.random.normal(center_mean, center_std, (n_boxes, 3)),
        np.abs(np.random.normal(size_mean, size_std, (n_boxes, 3))),
        np.random.uniform(.0, 2 * np.pi, (n_boxes, 1))
    ), axis=1)
    initial_boxes[:, 3: -1] = np.log(initial_boxes[:, 3: -1])
    initial_boxes = torch.tensor(initial_boxes, dtype=torch.float, device=device)
    boxes = initial_boxes.clone().detach()
    boxes.requires_grad = True

    initial_scores = np.random.random((n_boxes, n_classes + 1))
    initial_scores[:, -1] = .0
    initial_scores = torch.tensor(initial_scores, dtype=torch.float, device=device)
    scores = initial_scores.clone().detach()
    scores.requires_grad = True

    for i in range(len(predicted['boxes'])):
        predicted['boxes'][i] = torch.tensor(predicted['boxes'][i], dtype=torch.float, device=device)
        predicted['labels'][i] = torch.tensor(predicted['labels'][i], dtype=torch.long, device=device)

    optimizer = torch.optim.Adam([boxes, scores], lr=.01)
    for i in range(500):
        i_loss = torch.tensor(.0, dtype=torch.float, device=device)
        for j in range(len(predicted['boxes'])):
            optimizer.zero_grad()
            loss, rows = match_boxes(predicted['boxes'][j], predicted['labels'][j], boxes, scores)
            visible_index = check_boxes_in_camera(boxes.detach().cpu().numpy(), predicted['cameras'][j])
            no_object_index = np.ones(n_boxes, dtype=np.bool)
            no_object_index[rows] = False
            no_object_index = np.logical_and(visible_index, no_object_index)
            n_no_object = np.sum(no_object_index)
            no_object_nll = torch.nn.functional.cross_entropy(
                scores[no_object_index],
                torch.ones(int(n_no_object), dtype=torch.long, device=scores.device) * n_classes,
                reduction='none'
            )
            n_matched = len(rows)
            loss = torch.sum(loss) + torch.sum(no_object_nll) * no_object_weight
            loss = loss / max(n_matched + n_no_object, 1)
            i_loss += loss
        i_loss = i_loss / len(predicted['boxes'])
        i_loss.backward()
        optimizer.step()
        print('i:', i, 'loss:', i_loss.detach().cpu().numpy())
    boxes = boxes.detach().cpu().numpy()
    boxes[:, 3: -1] = np.exp(boxes[:, 3: -1])
    scores = torch.softmax(scores, dim=1).detach().cpu().numpy()
    return {
        'boxes': boxes,
        'labels': np.argmax(scores, axis=1),
        'scores': np.max(scores, axis=1)
    }


def run_layout_experiment():
    true_layout, layouts = generate_layout_dataset(n=4, center_threshold=.3, size_threshold=.3, angle_threshold=.3)
    print('layouts:', layouts)
    layout = optimize_layout(layouts)
    print('true layout:', true_layout)
    print('layout:', layout)


def run_box_experiment():
    n = 10
    n_boxes = 4
    n_classes = 10
    true, predicted = generate_box_dataset(
        n=n, n_boxes=n_boxes, n_classes=n_classes, center_std=.2, size_mean=.05, size_std=.02,
        class_probability=.1, drop_probability=.2, center_threshold=.02, size_threshold=.3, angle_threshold=.3
    )
    cameras = generate_camera_dataset(n=n, angle_threshold=.3)
    predicted['cameras'] = cameras
    update_box_dataset_with_cameras(predicted)
    print('predicted boxes:')
    for i in range(len(predicted['boxes'])):
        print('box set:')
        for j in range(len(predicted['boxes'][i])):
            print(predicted['boxes'][i][j], '|', predicted['labels'][i][j], '|', predicted['scores'][i][j])
    # TODO: + 10 here is for complication of current experiments
    optimized = optimize_boxes(predicted, n_boxes=n_boxes + 10, n_classes=n_classes, no_object_weight=.4)
    print('true boxes:')
    for i in range(len(true['boxes'])):
        print(true['boxes'][i], '|', true['labels'][i])
    print('boxes:')
    for i in range(len(optimized['boxes'])):
        print(optimized['boxes'][i], '|', optimized['labels'][i], '|', optimized['scores'][i])


if __name__ == '__main__':
    np.set_printoptions(precision=5, suppress=True, sign=' ')
    run_box_experiment()
