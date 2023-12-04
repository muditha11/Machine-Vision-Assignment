import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
from .constants import classes_label_map


class OxfordPetCustom(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        self.imgs = [im for im in os.listdir(self.main_dir) if im.endswith(".jpg")]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.imgs[idx])
        label = classes_label_map[
            "_".join(self.imgs[idx].split(".jpg")[0].split("_")[:-1])
        ]

        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, label
