import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

try:
    from .iou import calculate_2d_giou, calculate_3d_giou
except:
    print('WARNING!!! Install cuda ops!')


class BoxMatchingLoss:
    @staticmethod
    def prepare_repeated(
            seen_boxes,
            seen_labels,
            boxes,
            scores,
        ):
        n_boxes = len(boxes)
        n_seen_boxes = len(seen_boxes)
        if not n_seen_boxes:
            return torch.tensor(0.0, dtype=torch.float, device=boxes.device), []

        to_concat = [
            boxes[:, :3],
            torch.exp(boxes[:, 3: -1]),
            boxes[:, -1:],
        ]
        exp_boxes = torch.cat(to_concat, dim=1)
        repeated_boxes = exp_boxes.repeat_interleave(n_seen_boxes, 0)
        repeated_scores = scores.repeat_interleave(n_seen_boxes, 0)
        repeated_seen_boxes = seen_boxes.repeat(n_boxes, 1)
        repeated_seen_labels = seen_labels.repeat(n_boxes)

        repeated = {
            'boxes': repeated_boxes,
            'scores': repeated_scores,
            'seen_boxes': repeated_seen_boxes,
            'seen_labels': repeated_seen_labels,
        }

        return repeated

    @staticmethod
    def calculate_3d(
            n_boxes: int,
            n_seen_boxes: int,
            repeated_boxes,
            repeated_scores,
            repeated_seen_boxes,
            repeated_seen_labels,
            giou_coef: float = 0.5,
            nll_coef: float = 0.5,
            l1_coef: float = 0.0,
        ):
        pairwise_giou, _ = calculate_3d_giou(
            box3d1=repeated_boxes[None, ...],
            box3d2=repeated_seen_boxes[None, ...],
        )
        pairwise_giou = pairwise_giou.reshape(n_boxes, n_seen_boxes)
        pairwise_l1 = torch.mean(
            torch.abs(repeated_boxes[:, :3] - repeated_seen_boxes[:, :3]),
            dim=1,
        )
        pairwise_l1 = pairwise_l1.reshape(n_boxes, n_seen_boxes)
        pairwise_nll = F.cross_entropy(
            repeated_scores,
            repeated_seen_labels,
            reduction='none',
        )
        pairwise_nll = pairwise_nll.reshape(n_boxes, n_seen_boxes)
        cost = (
            giou_coef * pairwise_giou +
            nll_coef * pairwise_nll +
            l1_coef * pairwise_l1
        )
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

        return cost[rows, columns], rows

    @staticmethod
    def calculate_2d(
            n_boxes: int,
            n_seen_boxes: int,
            repeated_boxes,
            repeated_scores,
            repeated_seen_boxes,
            repeated_seen_labels,
            giou_coef: float = 0.5,
            nll_coef: float = 0.5,
            l1_coef: float = 0.0,
        ):
        boxes_projections = Camera.project_boxes_onto_camera_plane(
            boxes=repeated_boxes,
            camera=camera,
            mode='minmax',
        )
        seen_boxes_projections = Camera.project_boxes_onto_camera_plane(
            boxes=repeated_seen_boxes,
            camera=camera,
            mode='minmax',
        )

        pairwise_giou, _ = calculate_2d_giou(
            box1=boxes_projections[None, ...],
            box2=seen_boxes_projections[None, ...],
        )
        pairwise_giou = pairwise_giou.reshape(n_boxes, n_seen_boxes)
        '''
        TODO projections
        pairwise_l1 = torch.mean(
            torch.abs(repeated_boxes[:, :3] - repeated_seen_boxes[:, :3]),
            dim=1,
        )
        pairwise_l1 = pairwise_l1.reshape(n_boxes, n_seen_boxes)
        '''
        pairwise_l1 = 0
        pairwise_nll = F.cross_entropy(
            repeated_scores,
            repeated_seen_labels,
            reduction='none',
        )
        pairwise_nll = pairwise_nll.reshape(n_boxes, n_seen_boxes)
        cost = (
            giou_coef * pairwise_giou +
            nll_coef * pairwise_nll +
            l1_coef * pairwise_l1
        )
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

        return cost[rows, columns], rows
