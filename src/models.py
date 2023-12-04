import torch.nn as nn


class ClassificationModel(nn.Module):
    def __init__(self, backbone, mode="encoder"):
        super().__init__()
        self.backbone = backbone

        if mode == "pretrained_encoder":
            self.backbone.fc = nn.Identity()
            for param in backbone.parameters():
                param.requires_grad = False

        elif mode == "tune_dec":
            self.backbone.fc = nn.Sequential(
                nn.Linear(2048, 1000), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1000, 37)
            )
            for param in backbone.parameters():
                param.requires_grad = False
            for param in backbone.fc.parameters():
                param.requires_grad = True

        elif mode == "tune_enc_dec":
            self.backbone.fc = nn.Sequential(
                nn.Linear(2048, 1000), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(1000, 37)
            )
            for param in backbone.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return x
