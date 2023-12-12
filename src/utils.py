import numpy as np
import matplotlib.pyplot as plt
from .constants import classes
from sklearn.metrics import confusion_matrix
import seaborn as sns
# from torch.utils.tensorboard import SummaryWriter
# import logging
# import importlib


# class Logger:
#     def __init__(self) -> None:
#         self.logger = logging.getLogger()
#         self.logger.setLevel(logging.root.level)
#         self.display_info = self.logger.level <= logging.INFO
#         self.plotter = None
#         self.data = {}

#     # logging
#     def info(self, txt: str) -> None:
#         self.logger.info(txt)

#     def warn(self, txt: str) -> None:
#         self.logger.warn(txt)

#     def error(self, txt: str) -> None:
#         self.logger.error(txt)

#     # plotting
#     def init_plotter(self, logdir):
#         self.plotter = SummaryWriter(logdir)

#     def step(self, epoch):
#         for k, v in self.data.items():
#             v = [val for val in v if val is not None]
#             if len(v) != 0:
#                 self.plotter.add_scalar(k, sum(v) / len(v), epoch)
#         self.data = dict()

#     def _accumulate(self, name, y):
#         if name in self.data.keys():
#             self.data[name].append(y)
#         else:
#             self.data[name] = [y]

#     def accumulate_train_loss(self, y) -> None:
#         self._accumulate("Loss/train", y)

#     def accumulate_val_loss(self, y) -> None:
#         self._accumulate("Loss/validation", y)

#     def accumulate_analysis(self, name, y) -> None:
#         self._accumulate(f"Analysis/{name}", y)


def visualize_batch(batch):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    num_rows = 8
    num_cols = len(batch[1][0]) // 8

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 18))

    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            img = batch[1][0][index].numpy().transpose((1, 2, 0)) * std + mean
            img = np.clip(img, 0, 1)
            axes[i, j].imshow(img)
            class_num = batch[1][1][index].item()
            axes[i, j].set_title(f"{classes[class_num]}-{class_num}")
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y, yhat):
    cm = confusion_matrix(y, yhat)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=np.unique(y),
        yticklabels=np.unique(y),
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# def load_class(target):
#     """loads a class using a target"""
#     *module_name, class_name = target.split(".")
#     module_name = ".".join(module_name)
#     module = importlib.import_module(module_name)
#     cls = getattr(module, class_name)
#     return cls


# # loads scientific notations as float
# yaml_loader = yaml.SafeLoader
# yaml_loader.add_implicit_resolver(
#     "tag:yaml.org,2002:float",
#     re.compile(
#         """^(?:
#      [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
#     |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
#     |\\.[0-9_]+(?:[eE][-+][0-9]+)?
#     |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
#     |[-+]?\\.(?:inf|Inf|INF)
#     |\\.(?:nan|NaN|NAN))$""",
#         re.X,
#     ),
#     list("-+0123456789."),
# )
