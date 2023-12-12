import os
from typing import Any
from src.utils import init_obj_from_conf, plot_confusion_matrix
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from omegaconf import OmegaConf
from torch import nn
from tqdm.auto import tqdm
import numpy as np


class Evaluator:
    def _load_data(self, mock_batch_count: int = -1):
        ds = init_obj_from_conf(self.conf.data, split="test")
        if mock_batch_count > 0:
            ds = Subset(
                ds,
                range(int(mock_batch_count * self.conf.test.loader_params.batch_size)),
            )
        self.dl: DataLoader = DataLoader(ds, **dict(self.conf.test.loader_params))

    def _load_model(self, weight_path: str):
        # model
        self.model: nn.Module = init_obj_from_conf(self.conf.model, device=self.device)
        self.model.load_weights(weight_path)

    def __init__(
        self, config_path, out_dir, weight_path, device, mock_batch_count=-1
    ) -> None:
        with open(config_path) as handler:
            conf = OmegaConf.create(yaml.load(handler, yaml.FullLoader))
        os.makedirs(out_dir, exist_ok=True)

        self.conf = conf
        self.device = device
        self.out_dir = out_dir

        self._load_data(mock_batch_count)
        self._load_model(weight_path)

    def __call__(self) -> Any:
        preds = []
        labels = []
        with torch.no_grad():
            for batch in tqdm(self.dl):
                info = self.model(batch)
                logits = info["logits"]
                label = batch[1]
                pred = logits.argmax(1)
                preds.extend(pred.tolist())
                labels.extend(label.tolist())
        preds, labels = np.array(preds), np.array(labels)

        plot_confusion_matrix(
            labels, preds, os.path.join(self.out_dir, "confusion-matrix.png")
        )
        acc = sum(preds == labels) / len(preds)
        report = f"Accuracy: {acc}"
        with open(os.path.join(self.out_dir, "report.txt"), "w") as handler:
            handler.write(report)

        print(report)
