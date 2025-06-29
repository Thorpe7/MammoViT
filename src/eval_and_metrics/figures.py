import matplotlib.pyplot as plt
import pandas as pd
import json

def plot_loss_curve(metrics_path, output_path):
    """
    Generate a loss curve showing training and validation loss over epochs.
    """
    df = pd.read_csv(metrics_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", marker='o')
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "loss_curve.png")
    plt.close()

def plot_accuracy_curve(metrics_path, output_path):
    """
    Generate an accuracy curve showing training and validation accuracy over epochs.
    """
    df = pd.read_csv(metrics_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_acc"], label="Training Accuracy", marker='o')
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "accuracy_curve.png")
    plt.close()

def plot_confusion_matrix(cm_path, output_path, class_names):
    """
    Generate a confusion matrix plot.
    """
    with open(cm_path, "r") as f:
        data = json.load(f)
    cm = data["confusion_matrix"]

    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i][j], horizontalalignment="center", color="white" if cm[i][j] > cm.max() / 2 else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png")
    plt.close()
