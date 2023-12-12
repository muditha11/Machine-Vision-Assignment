from typing import Sequence, Dict
import torch
from torch.nn import CrossEntropyLoss as TorchCrossEntropyLoss


class CrossEntropyLoss:
    def __init__(self, device, logger) -> None:
        self.logger = logger
        self.device = device
        self.loss_fn = TorchCrossEntropyLoss()

    def __call__(
        self, info: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        logits = info["logits"]
        labels = batch[1].to(self.device)
        loss = self.loss_fn(logits, labels)
        return loss
