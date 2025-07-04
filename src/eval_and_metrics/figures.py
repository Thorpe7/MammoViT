# Retry without torch dependency since we only need numpy and matplotlib

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns
import numpy as np


def plot_accuracy(metrics_df):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df["epoch"], metrics_df["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(metrics_df["epoch"], metrics_df["val_acc"], label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_loss(metrics_df):
    plt.figure(figsize=(10, 5))
    plt.plot(metrics_df["epoch"], metrics_df["train_loss"], label="Train Loss", marker='o')
    plt.plot(metrics_df["epoch"], metrics_df["val_loss"], label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(json_path):
    with open(json_path, "r") as f:
        cm_data = json.load(f)
    cm = np.array(cm_data["confusion_matrix"])
    labels = cm_data.get("labels", [str(i) for i in range(len(cm))])

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

def plot_class_counts(dir_path: Path, normalize_vals: bool):
    """Grabs class counts and plots them.

    Args:
        dir_path (PosixPath): Path to top level data directory
    """

    labels = []
    counts = []
    pointer = 0
    for dir in dir_path.iterdir():
        if pointer >= len(os.listdir()):
            break
        file_count = 0
        labels.append(dir.name)
        for file in dir.glob("*.png"):
            if file.is_file:
                file_count += 1
        counts.append(file_count)
    print(labels)
    print(counts)
    if normalize_vals:
        counts = normalize_values(counts)
    test = plt.bar(counts, counts, 0.1)
    plt.bar_label(test, labels)
    plt.show()

def normalize_values(input_list: list):
    input_arr = np.array(input_list)
    min_val = min(input_arr)
    max_val = max(input_arr)
    normalized = (input_arr - min_val) / (max_val - min_val)
    normalized = list(normalized)
    return normalized
if __name__ == "__main__":
    data_dir = Path("/Users/thorpe/git_repos/MammoViT/data/KAU")
    plot_class_counts(data_dir, True)
    data_dir2 = Path("/Users/thorpe/git_repos/MammoViT/data/INbreast/OrganizedByBiRads_PNG")
    plot_class_counts(data_dir2, True)
