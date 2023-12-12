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
import matplotlib.pyplot as plt


class Evaluator:
    def _load_data(self, mock_batch_count: int = -1):
        ds = init_obj_from_conf(self.conf.data, split="test")
        if mock_batch_count > 0:
            ds = Subset(
                ds,
                range(int(mock_batch_count * self.conf.test.loader_params.batch_size)),
            )
        params = dict(self.conf.test.loader_params)
        params["shuffle"] = True
        self.dl: DataLoader = DataLoader(ds, **params)

    def _load_model(self, weight_path: str):
        # model
        self.model: nn.Module = init_obj_from_conf(self.conf.model, device=self.device)
        self.model.load_weights(weight_path)

    def __init__(
        self, config_path, out_dir, weight_path, device, mock_batch_count=-1
    ) -> None:
        with open(config_path) as handler:
            conf = OmegaConf.create(yaml.load(handler, yaml.FullLoader))
        os.makedirs(os.path.join(out_dir, "results"), exist_ok=True)

        self.conf = conf
        self.device = device
        self.out_dir = out_dir

        self._load_data(mock_batch_count)
        self._load_model(weight_path)

    def _set_color(self, ax, color):
        for pos in ["bottom", "top", "right", "left"]:
            ax.spines[pos].set_color(color)
            ax.spines[pos].set_linewidth(3)

    def _save_grid(self, path, img_buffer, truth_buffer, title_buffer):
        grid_size = self.conf.test.visualizer_params.grid_size
        fig, ax = plt.subplots(
            grid_size, grid_size, figsize=(2 * grid_size, 2 * grid_size)
        )
        plt.suptitle(f"Prediction results (label:prediction)")
        break_through = False
        for i in range(grid_size):
            for j in range(grid_size):
                ax[i][j].set_xticks([])
                ax[i][j].set_yticks([])

        for i in range(grid_size):
            for j in range(grid_size):
                if len(img_buffer) == 0:
                    break_through = True
                    break
                img = img_buffer.pop(0)
                truth = truth_buffer.pop(0)
                title = title_buffer.pop(0)
                ax[i][j].imshow(img)
                ax[i][j].set_title(title)
                if truth:
                    self._set_color(ax[i][j], "green")
                else:
                    self._set_color(ax[i][j], "red")
            if break_through:
                break
        plt.close()
        fig.savefig(path)

    def __call__(self) -> Any:
        preds = []
        labels = []

        img_buffer = []
        truth_buffer = []
        title_buffer = []
        max_fig_count = self.conf.test.visualizer_params.max_fig_count
        grid_id = 0

        with torch.no_grad():
            for batch in tqdm(self.dl):
                info = self.model(batch)
                logits = info["logits"]
                label = batch[1]
                pred = logits.argmax(1)
                preds.extend(pred.tolist())
                labels.extend(label.tolist())

                if grid_id < max_fig_count:
                    imgs = [*batch[0].numpy().transpose(0, 2, 3, 1)]
                    label, pred = label.numpy(), pred.detach().cpu().numpy()
                    truth = label == pred
                    title = [f"{l}:{p}" for (l, p) in zip(label, pred)]

                    img_buffer.extend(imgs)
                    truth_buffer.extend(truth)
                    title_buffer.extend(title)
                    while (
                        len(img_buffer)
                        > self.conf.test.visualizer_params.grid_size**2
                        and grid_id < max_fig_count
                    ):
                        self._save_grid(
                            os.path.join(
                                self.out_dir, "results", f"result-{grid_id}.jpg"
                            ),
                            img_buffer,
                            truth_buffer,
                            title_buffer,
                        )
                        grid_id += 1
            if len(img_buffer) > 0 and grid_id < max_fig_count:
                self._save_grid(
                    os.path.join(self.out_dir, "results", f"result-{grid_id}.jpg"),
                    img_buffer,
                    truth_buffer,
                    title_buffer,
                )

        preds, labels = np.array(preds), np.array(labels)

        plot_confusion_matrix(
            labels, preds, os.path.join(self.out_dir, "confusion-matrix.png")
        )
        acc = sum(preds == labels) / len(preds)
        report = f"Accuracy: {acc}"
        with open(os.path.join(self.out_dir, "report.txt"), "w") as handler:
            handler.write(report)

        print(report)
