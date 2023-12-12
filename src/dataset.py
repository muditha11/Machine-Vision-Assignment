import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from .constants import classes_label_map, classes, img_wh
from torchvision.datasets import OxfordIIITPet


class OxfordPetCustom(Dataset):
    valid_splits = ["train", "val", "test"]

    def _download(self, root: str):
        pytorch_ds_root = root.rstrip("oxford-iiit-pet")
        OxfordIIITPet(pytorch_ds_root, download=True)
        os.remove(
            os.path.join(pytorch_ds_root, "oxford-iiit-pet", "annotations.tar.gz")
        )
        os.remove(os.path.join(pytorch_ds_root, "oxford-iiit-pet", "images.tar.gz"))

    def __init__(self, root="./data/oxford-iiit-pet", split="train"):
        assert root.endswith("oxford-iiit-pet")

        if split not in self.valid_splits:
            raise ValueError(
                f"Invalid split definition. Valid splits are {self.valid_splits}"
            )
        if not os.path.exists(root):
            self._download(root)

        self.root = root
        self.split = split

        with open(f"{root}/annotations/splits/" + split + ".txt", "r") as file:
            read_list = [line.strip() for line in file.readlines()]
        self.imgs = read_list

        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_wh[::-1], scale=[0.8, 1.0]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

        self.transform_test = transforms.Compose(
            [
                transforms.Resize(img_wh[::-1]),
                transforms.ToTensor(),
            ]
        )
        self.img_dir = os.path.join(root, "images")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.imgs[idx])
        label = classes_label_map[
            "_".join(self.imgs[idx].split(".jpg")[0].split("_")[:-1])
        ]

        image = Image.open(img_loc).convert("RGB")
        if self.split in ["test", "val"]:
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
