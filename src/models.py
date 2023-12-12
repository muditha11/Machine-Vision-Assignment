import torch
import torch.nn as nn
import torchvision
from typing import Sequence, Dict


# TODO: drop the usage of this model
class ClassificationModel(nn.Module):
    def __init__(self, backbone, mode="no_tun-enc"):
        super().__init__()
        self.backbone = backbone

        if mode == "no_tun-enc":
            self.backbone.fc = nn.Identity()
            for param in backbone.parameters():
                param.requires_grad = False

        elif mode == "tune-dec":
            self.backbone.fc = nn.Sequential(
                nn.Linear(2048, 1000), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1000, 37)
            )
            for param in backbone.parameters():
                param.requires_grad = False
            for param in backbone.fc.parameters():
                param.requires_grad = True

        elif mode == "tune-enc_dec":
            self.backbone.fc = nn.Sequential(
                nn.Linear(2048, 1000), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1000, 37)
            )
            for param in backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return x


class Classifier(nn.Module):
    head_end = 37
    head_starts = {"ResNet50": 2048}
    valid_encs = ["ResNet50"]

    def _init_head(self, head_inters, use_batch_norm):
        head_dims = [self.head_starts[self.enc_name], *head_inters, self.head_end]
        head_linear_layers = [
            nn.Linear(head_dims[i], head_dims[i + 1]) for i in range(len(head_dims) - 1)
        ]
        head_layers = [
            layer
            for triplet in zip(
                head_linear_layers,
                [nn.BatchNorm1d(i) for i in head_dims[1:]],
                [nn.ReLU()] * (len(head_linear_layers) - 1),
            )
            for layer in triplet
        ]
        head_layers.extend([head_linear_layers[-1]])
        self.decoder = nn.Sequential(*head_layers)

    def __init__(
        self,
        enc_name: str = "ResNet50",
        head_inters: Sequence[int] = [],
        drop_out: float = 0.2,
        use_batch_norm: bool = False,
        device: int | str = 0,
    ) -> None:
        if enc_name not in self.valid_encs:
            raise ValueError(
                f"Unsupported 'enc_name' definition. Supported encoders are {self.valid_encs}"
            )
        super().__init__()
        self.enc_name = enc_name
        self.device = device
        self.drop_out = drop_out

        if enc_name == "ResNet50":
            self.encoder = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT
            )
            self.encoder.fc = nn.Identity()

        self._init_head(head_inters, use_batch_norm)

        self.to(device)

    def forward(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        imgs, _ = batch
        imgs = imgs.to(self.device)
        embs = self.encoder(imgs)
        logits = self.decoder(embs)
        return {"logits": logits}

    def load_weights(self, weights_path: str) -> None:
        conf = torch.load(weights_path)
        sd = conf["state"]["model"]
        self.load_state_dict(sd)
