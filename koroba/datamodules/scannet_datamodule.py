from pathlib import Path

import torch

from koroba.datamodules import BaseDataModule


class ScannetDataModule(BaseDataModule):
    def __init__(
            scan_path: Path('./data/scans/scene0000_00/instance-filt/999.png'),
            device: torch.device = torch.device('cpu'),
        ):
        self.scan_path = scan_path
        self.device = device

    def get_bounding_boxes_from_instances(
            self,
            frame: np.ndarray,
        ):
        return

    def a(
            self,
        ):
        for frame in frames:
            self.get_bounding_boxes_from_instances(frame=frame)

        return result

    def setup(self):
        self.true = None
        frames = None
        self.seen = self.get_boxes2d():

        initial_boxes = \
            torch.cat(tuple(filter(lambda x: len(x), self.seen['boxes'])))
        center_mean = initial_boxes[:, :3].mean(axis=0)
        center_std = initial_boxes[:, :3].std(axis=0)
        size_mean = initial_boxes[:, 3:-1].mean(axis=0)
        size_std = initial_boxes[:, 3:-1].std(axis=0)

        to_concat = (
            torch.tensor(
                np.random.normal(
                    center_mean.cpu().numpy(),
                    center_std.cpu().numpy(),
                    (self.n_boxes, 3),
                ),
                dtype=torch.float,
            ),
            torch.tensor(
                np.abs(np.random.normal(
                    size_mean.cpu().numpy(),
                    size_std.cpu().numpy(),
                    (self.n_boxes, 3),
                )),
                dtype=torch.float,
            ),
            torch.tensor(
                np.random.uniform(0.0, 2 * np.pi, (self.n_boxes, 1)),
                dtype=torch.float,
            ),
        )
        initial_boxes = torch.cat(to_concat, axis=1)
        initial_boxes[:, 3:-1] = torch.log(initial_boxes[:, 3:-1])
        initial_boxes = \
            torch.tensor(initial_boxes, dtype=torch.float, device=self.device)
        optimized_boxes = initial_boxes.clone().detach()
        optimized_boxes.requires_grad = True

        initial_scores = np.random.random((self.n_boxes, self.n_classes + 1))
        initial_scores[:, -1] = 0.0
        initial_scores = \
            torch.tensor(initial_scores, dtype=torch.float, device=self.device)
        optimized_scores = initial_scores.clone().detach()
        optimized_scores.requires_grad = True

        self.optimized = {
            'boxes': optimized_boxes,
            'scores': optimized_scores,
        }
