from sklearn.metrics import confusion_matrix
import json
import pandas as pd

def initialize_metric_logs(log_dir):
    """
    Initialize metric logs at the beginning of fine-tuning.
    Creates CSV files for tracking loss and accuracy, and JSON for confusion matrix data.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # CSV for epoch-wise metrics
    metrics_path = log_dir / "epoch_metrics.csv"
    if not metrics_path.exists():
        df = pd.DataFrame(columns=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])
        df.to_csv(metrics_path, index=False)

    # JSON for confusion matrix
    cm_path = log_dir / "confusion_matrix.json"
    if not cm_path.exists():
        with open(cm_path, "w") as f:
            json.dump({}, f)


def log_epoch_metrics(log_dir, epoch, train_loss, val_loss, train_acc, val_acc):
    """
    Append training/validation metrics for the current epoch to CSV.
    """
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    metrics_path = log_dir / "epoch_metrics.csv"
    new_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc
    }
    df = pd.DataFrame([new_row])
    df.to_csv(metrics_path, mode='a', header=not metrics_path.exists(), index=False)


def save_confusion_matrix(log_dir, labels, predictions, num_classes, class_labels):
    """
    Save confusion matrix as JSON (class x class matrix) with class labels.
    """
    log_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    cm = confusion_matrix(labels, predictions, labels=range(num_classes)).tolist()  # Ensure correct number of classes
    cm_path = log_dir / "confusion_matrix.json"
    with open(cm_path, "w") as f:
        json.dump({
            "confusion_matrix": cm,
            "class_labels": class_labels  # Include class labels for clarity
        }, f)
