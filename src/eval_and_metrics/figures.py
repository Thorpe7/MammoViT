# Retry without torch dependency since we only need numpy and matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns
import numpy as np

# Plotting function for accuracy
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

# Plotting function for loss
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

# Load and plot confusion matrix from JSON
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


