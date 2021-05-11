import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from koroba.utils import Camera

try:
    from .iou import calculate_2d_giou, calculate_3d_giou
except:
    print('WARNING!!! Install cuda ops!')


class BoxMatchingLoss:
    def __init__(
            self,
            giou_coef: float = 0.5,
            nll_coef: float = 0.5,
            l1_coef: float = 0.0,
        ):
        self.giou_coef = giou_coef
        self.nll_coef = nll_coef
        self.l1_coef = l1_coef

    @staticmethod
    def prepare_repeated_boxes(
            optimized_boxes,
            optimized_scores,
            seen_boxes,
            seen_labels,
        ):
        n_optimized = len(optimized_boxes)
        n_seen = len(seen_boxes)

        repeated_optimized_boxes = optimized_boxes.repeat_interleave(n_seen, 0)
        repeated_optimized_scores = \
            optimized_scores.repeat_interleave(n_seen, 0)
        repeated_seen_boxes = seen_boxes.repeat(n_optimized, 1)
        repeated_seen_labels = seen_labels.repeat(n_optimized)

        repeated = {
            'optimized_boxes': repeated_optimized_boxes,
            'optimized_scores': repeated_optimized_scores,
            'seen_boxes': repeated_seen_boxes,
            'seen_labels': repeated_seen_labels,
        }

        return repeated

    @staticmethod
    def prepare_repeated_projections(
            optimized_projections,
            optimized_scores,
            seen_projections,
            seen_labels,
        ):
        n_optimized = len(optimized_projections)
        n_seen = len(seen_projections)
        repeated_optimized_projections = \
            optimized_projections.repeat_interleave(n_seen, 0)
        repeated_optimized_scores = \
            optimized_scores.repeat_interleave(n_seen, 0)
        repeated_seen_projections = seen_projections.repeat(n_optimized, 1)
        repeated_seen_labels = seen_labels.repeat(n_optimized)

        repeated = {
            'optimized_projections': repeated_optimized_projections,
            'optimized_scores': repeated_optimized_scores,
            'seen_projections': repeated_seen_projections,
            'seen_labels': repeated_seen_labels,
        }

        return repeated

    def calculate_3d(
            self,
            n_boxes: int,
            n_seen_boxes: int,
            repeated_optimized_boxes: Tensor,
            repeated_optimized_scores: Tensor,
            repeated_seen_boxes: Tensor,
            repeated_seen_labels: Tensor,
        ):
        pairwise_giou, _ = calculate_3d_giou(
            box3d1=repeated_optimized_boxes[None, ...],
            box3d2=repeated_seen_boxes[None, ...],
        )
        pairwise_giou = pairwise_giou.reshape(n_boxes, n_seen_boxes)
        pairwise_l1 = torch.mean(
            torch.abs(repeated_optimized_boxes[:, :3] - repeated_seen_boxes[:, :3]),
            dim=1,
        )
        pairwise_l1 = pairwise_l1.reshape(n_boxes, n_seen_boxes)
        pairwise_nll = F.cross_entropy(
            repeated_optimized_scores,
            repeated_seen_labels,
            reduction='none',
        )
        pairwise_nll = pairwise_nll.reshape(n_boxes, n_seen_boxes)
        cost = (
            self.giou_coef * pairwise_giou +
            self.nll_coef * pairwise_nll +
            self.l1_coef * pairwise_l1
        )
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

        return cost[rows, columns], rows

    def calculate_2d(
            self,
            n_projections: int,
            n_seen_projections: int,
            repeated_optimized_projections: Tensor,
            repeated_optimized_scores: Tensor,
            repeated_seen_projections: Tensor,
            repeated_seen_labels: Tensor,
            camera: Tensor,
        ):
        pairwise_giou, _ = calculate_2d_giou(
            box1=repeated_optimized_projections[None, ...],
            box2=repeated_seen_projections[None, ...],
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
            repeated_optimized_scores,
            repeated_seen_labels,
            reduction='none',
        )
        pairwise_nll = pairwise_nll.reshape(n_projections, n_seen_projections)
        cost = (
            self.giou_coef * pairwise_giou +
            self.nll_coef * pairwise_nll +
            self.l1_coef * pairwise_l1
        )
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())

        return cost[rows, columns], rows
