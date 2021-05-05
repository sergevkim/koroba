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
        ):
        pairwise_giou, _ = calculate_3d_giou(
            box3d1=repeated_boxes[None, ...],
            box3d2=repeated_seen_boxes[None, ...],
        )
        pairwise_giou = pairwise_giou.reshape(n_boxes, n_seen_boxes)
        # pairwise_l1 = torch.mean(torch.abs(repeated_boxes[:, :3] - repeated_p_boxes[:, :3]), dim=1)
        # pairwise_l1 = pairwise_l1.reshape(n_boxes, n_p_boxes)
        pairwise_nll = F.cross_entropy(
            repeated_scores,
            repeated_seen_labels,
            reduction='none',
        )
        pairwise_nll = pairwise_nll.reshape(n_boxes, n_seen_boxes)
        cost = pairwise_giou + pairwise_nll
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
        ):
        #TODO projections
        pairwise_giou, _ = calculate_2d_giou(
            box1=repeated_boxes[None, ...],
            box2=repeated_seen_boxes[None, ...],
        )
        #end of TODO
        pairwise_giou = pairwise_giou.reshape(n_boxes, n_seen_boxes)
        # pairwise_l1 = torch.mean(torch.abs(repeated_boxes[:, :3] - repeated_p_boxes[:, :3]), dim=1)
        # pairwise_l1 = pairwise_l1.reshape(n_boxes, n_p_boxes)
        pairwise_nll = F.cross_entropy(
            repeated_scores,
            repeated_seen_labels,
            reduction='none',
        )
        pairwise_nll = pairwise_nll.reshape(n_boxes, n_seen_boxes)
        cost = pairwise_giou + pairwise_nll
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

        return cost[rows, columns], rows
