import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
from .constants import classes_label_map, mean, std, classes


class OxfordPetCustom(Dataset):
    def __init__(self, main_dir, split):
        self.main_dir = main_dir
        self.split = split

        with open(
            "./data/oxford-iiit-pet/annotations/splits/" + split + ".txt", "r"
        ) as file:
            read_list = [line.strip() for line in file.readlines()]
        self.imgs = read_list

        self.transform_train = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.RandomCrop(128, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.imgs[idx])
        label = classes_label_map[
            "_".join(self.imgs[idx].split(".jpg")[0].split("_")[:-1])
        ]

        image = Image.open(img_loc).convert("RGB")
        if self.split == "test":
            tensor_image = self.transform_test(image)
        else:
            tensor_image = self.transform_train(image)
        return tensor_image, label


def save_split(my_list, split, save_root):
    with open(save_root + "/" + split + ".txt", "w") as file:
        for item in my_list:
            file.write(f"{item}\n")


def split_data(root, split=[0.7, 0.15, 0.15]):
    ims = os.listdir(root)
    train, val, test = [], [], []
    for i in classes:
        class_ims = []
        for j in ims:
            if j.startswith(i) and j.endswith(".jpg"):
                class_ims.append(j)
        t = int(len(class_ims) * split[0])
        v = t + int(len(class_ims) * split[1])
        train += class_ims[:t]
        val += class_ims[t:v]
        test += class_ims[v:]
    save_root = r"./data/oxford-iiit-pet/annotations/splits"
    save_split(train, "train", save_root), save_split(
        val, "val", save_root
    ), save_split(test, "test", save_root)
