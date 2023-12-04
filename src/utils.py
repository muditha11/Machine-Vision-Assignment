import numpy as np
import matplotlib.pyplot as plt
from .constants import classes
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
