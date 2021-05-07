import torch
from torch.optim import Optimizer

from koroba.datamodules import BaseDataModule


class Runner:
    def __init__(
            device: torch.device = torch.device('cpu'),
            max_epoch: int = 200,
            verbose: bool = False,
        ):
        self.device = device
        self.verbose = verbose

    def run_epoch(
            self,
            seen,
            optimized_boxes,
            optimized_scores,
            optimizer,
            epoch_idx: int,
        ):
        i_loss = torch.tensor(.0, dtype=torch.float, device=self.device)

        for j in range(len(seen['boxes'])):
            optimizer.zero_grad()

            seen_boxes = seen['boxes'][j]
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
                match_boxes_loss, rows = BoxMatchingLoss.calculate_3d(
                    n_boxes=len(optimized_boxes),
                    n_seen_boxes=len(seen_boxes),
                    repeated_boxes=repeated_boxes,
                    repeated_scores=repeated_scores,
                    repeated_seen_boxes=repeated_seen_boxes,
                    repeated_seen_labels=repeated_seen_labels,
                )
            else:
                match_boxes_loss, rows = BoxMatchingLoss.calculate_2d(
                    n_boxes=len(optimized_boxes),
                    n_seen_boxes=len(seen_boxes),
                    repeated_boxes=repeated_boxes,
                    repeated_scores=repeated_scores,
                    repeated_seen_boxes=repeated_seen_boxes,
                    repeated_seen_labels=repeated_seen_labels,
                    camera=camera,
                )

            visible_index = Camera.check_boxes_in_camera_fov(
                boxes=optimized_boxes.detach().cpu().numpy(),
                camera=camera,
            )
            no_object_index = np.ones(len(optimized_boxes), dtype=np.bool)
            no_object_index[rows] = False
            no_object_index = np.logical_and(visible_index, no_object_index)
            n_no_object = np.sum(no_object_index)
            no_object_nll = F.cross_entropy(
                optimized_scores[no_object_index],
                torch.ones(
                    int(n_no_object),
                    dtype=torch.long,
                    device=optimized_scores.device,
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

        i_loss = i_loss / len(seen['boxes'])
        i_loss.backward()
        optimizer.step()
        print(f'i: {i};  loss: {i_loss.detach().cpu().numpy()}')

    def run(
            self,
            n_boxes: int,
            datamodule: BaseDataModule,
            optimizer_name: Optimizer,
        ):
        initial_boxes = \
            np.concatenate(tuple(filter(lambda x: len(x), seen['boxes'])))
        center_mean = np.mean(initial_boxes[:, :3], axis=0)
        center_std = np.std(initial_boxes[:, :3], axis=0)
        size_mean = np.mean(initial_boxes[:, 3:-1], axis=0)
        size_std = np.std(initial_boxes[:, 3:-1], axis=0)

        to_concat = (
            np.random.normal(center_mean, center_std, (n_boxes, 3)),
            np.abs(np.random.normal(size_mean, size_std, (n_boxes, 3))),
            np.random.uniform(.0, 2 * np.pi, (n_boxes, 1)),
        )
        initial_boxes = np.concatenate(to_concat, axis=1)
        initial_boxes[:, 3: -1] = np.log(initial_boxes[:, 3: -1])
        initial_boxes = \
            torch.tensor(initial_boxes, dtype=torch.float, device=self.device)
        optimized_boxes = initial_boxes.clone().detach()
        optimized_boxes.requires_grad = True

        initial_scores = np.random.random((n_boxes, n_classes + 1))
        initial_scores[:, -1] = .0
        initial_scores = \
            torch.tensor(initial_scores, dtype=torch.float, device=self.device)
        scores = initial_scores.clone().detach()
        scores.requires_grad = True

        for i in range(len(seen['boxes'])):
            seen['boxes'][i] = torch.tensor(
                seen['boxes'][i],
                dtype=torch.float,
                device=self.device,
            )
            seen['labels'][i] = torch.tensor(
                seen['labels'][i],
                dtype=torch.long,
                device=self.device,
            )

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam([optimized_boxes, scores], lr=0.01)

        for epoch_idx in range(args.max_epoch):
            self.run_epoch(
                seen=seen,
                optimized_boxes=optimized_boxes,
                optimized_scores=optimized_scores,
                optimizer=optimizer,
                epoch_idx=epoch_idx,
            )

        optimized_boxes = optimized_boxes.detach().cpu().numpy()
        optimized_boxes[:, 3: -1] = np.exp(optimized_boxes[:, 3: -1])
        scores = torch.softmax(scores, dim=1).detach().cpu().numpy()
        optimized = {
            'boxes': optimized_boxes,
            'labels': np.argmax(scores, axis=1),
            'scores': np.max(scores, axis=1)
        }

        return optimized
