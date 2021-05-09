import numpy as np
import torch
import torch.nn.functional as F

from koroba.datamodules import BaseDataModule
from koroba.losses import BoxMatchingLoss
from koroba.utils import Camera


class Runner:
    def __init__(
            self,
            device: torch.device = torch.device('cpu'),
            max_epoch: int = 200,
            optimizer_name: str = 'adam',
            verbose: bool = False,
            giou_coef: float = 0.5,
            nll_coef: float = 0.5,
            l1_coef: float = 0.0,
            no_object_coef: float = 0.4,
        ):
        self.device = device
        self.max_epoch = max_epoch
        self.optimizer_name = optimizer_name
        self.verbose = verbose
        self.box_matching_criterion = BoxMatchingLoss(
            giou_coef=giou_coef,
            nll_coef=nll_coef,
            l1_coef=l1_coef,
        )
        self.no_object_coef = no_object_coef

    def run_epoch(
            self,
            seen,
            optimized_boxes,
            optimized_scores,
            constants,
            optimizer,
            epoch_idx: int,
            mode: str,
        ):
        i_loss = torch.tensor(.0, dtype=torch.float, device=self.device)

        for j, seen_boxes in enumerate(seen['boxes']):
            optimizer.zero_grad()

            #seen_boxes = seen['boxes'][j]
            seen_labels = seen['labels'][j]
            camera = seen['cameras'][j]

            repeated = BoxMatchingLoss.prepare_repeated(
                seen_boxes=seen_boxes,
                seen_labels=seen_labels,
                boxes=optimized_boxes,
                scores=optimized_scores,
            )
            repeated_boxes = repeated['boxes']
            repeated_scores = repeated['scores']
            repeated_seen_boxes = repeated['seen_boxes']
            repeated_seen_labels = repeated['seen_labels']

            if mode == '3d':
                box_matching_loss, rows = \
                        self.box_matching_criterion.calculate_3d(
                    n_boxes=len(optimized_boxes),
                    n_seen_boxes=len(seen_boxes),
                    repeated_boxes=repeated_boxes,
                    repeated_scores=repeated_scores,
                    repeated_seen_boxes=repeated_seen_boxes,
                    repeated_seen_labels=repeated_seen_labels,
                )
            else:
                box_matching_loss, rows = \
                        self.box_matching_criterion.calculate_2d(
                    n_boxes=len(optimized_boxes),
                    n_seen_boxes=len(seen_boxes),
                    repeated_boxes=repeated_boxes,
                    repeated_scores=repeated_scores,
                    repeated_seen_boxes=repeated_seen_boxes,
                    repeated_seen_labels=repeated_seen_labels,
                    camera=camera,
                )

            visible_index = Camera.check_boxes_in_camera_fov(
                boxes=optimized_boxes.detach(), #TODO remove detach
                camera=camera,
            )
            no_object_index = torch.ones(
                len(optimized_boxes),
                dtype=np.bool,
                device=visible_index.device,
            )
            no_object_index[rows] = False
            no_object_index = visible_index * no_object_index
            n_no_object = torch.sum(no_object_index)

            no_object_nll = F.cross_entropy(
                optimized_scores[no_object_index],
                torch.ones(
                    int(n_no_object),
                    dtype=torch.long,
                    device=optimized_scores.device,
                ) * constants['n_classes'],
                reduction='none',
            )
            n_matched = len(rows)
            loss = (
                torch.sum(box_matching_loss) +
                torch.sum(no_object_nll) * self.no_object_coef
            )
            loss = loss / max(n_matched + n_no_object, 1)
            i_loss += loss

        i_loss = i_loss / len(seen['boxes'])
        i_loss.backward()
        optimizer.step()
        print(f'epoch_idx: {epoch_idx};  loss: {i_loss.detach().cpu().numpy()}')

    def run(
            self,
            datamodule: BaseDataModule,
            mode: str,
        ):
        true = datamodule.get_true()
        seen = datamodule.get_seen()
        optimized = datamodule.get_optimized()
        constants = datamodule.get_constants()
        optimized_boxes = optimized['boxes']
        optimized_scores = optimized['scores']

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                params=[optimized_boxes, optimized_scores],
                lr=0.01,
            )

        for epoch_idx in range(self.max_epoch):
            self.run_epoch(
                seen=seen,
                optimized_boxes=optimized_boxes,
                optimized_scores=optimized_scores,
                constants=constants,
                optimizer=optimizer,
                epoch_idx=epoch_idx,
                mode=mode,
            )

        optimized_boxes = optimized_boxes.detach().cpu()
        optimized_boxes[:, 3:-1] = np.exp(optimized_boxes[:, 3:-1])
        optimized_scores = \
            torch.softmax(optimized_scores, dim=1).detach().cpu().numpy()

        optimized_result = {
            'boxes': optimized_boxes,
            'labels': np.argmax(optimized_scores, axis=1),
            'scores': np.max(optimized_scores, axis=1)
        }

        return optimized_result
