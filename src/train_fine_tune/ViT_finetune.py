import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import uuid
from itertools import product
from sklearn.metrics import accuracy_score
from pathlib import Path

from src.train_fine_tune.metric_collection import (
    initialize_metric_logs,
    log_epoch_metrics,
    save_confusion_matrix
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_partial_checkpoint(model_instance, optimizer, trial_index, epoch, hyperparams, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"partial_checkpoint_trial{trial_index}_epoch{epoch}.pth"
    torch.save({
        'trial_index': trial_index,
        'epoch': epoch,
        'hyperparams': hyperparams,
        'model_state_dict': model_instance.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)


def save_best_checkpoint(model_state, optimizer_state, best_params, epoch, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'params': best_params
    }, save_path)

    # Save best parameters separately as JSON
    params_path = save_path.parent / "best_params.json"
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)


def fine_tune_model_with_search(
    model_class,
    train_loader,
    val_loader,
    num_classes,
    tuner_epochs=50,
    run_final_train=True,
    optimize_params=True,
    logs_base_path="/content/drive/MyDrive/Embark_Labs/MammoViT/logs"
):
    run_id = str(uuid.uuid4())  # Each run to generate unique ID
    logs_base_path = Path(logs_base_path)
    base_ckpt_dir = logs_base_path / f"checkpoints/ViT_tuning/{run_id}"
    final_metrics_dir = logs_base_path / f"metrics/ViT_tuning/{run_id}"

    # Hyperparameter ranges for grid search
    lr_values = [0.01, 0.001, 0.0001]
    proj_dim_values = [32, 48, 64]
    n_blocks_values = [2, 3, 4, 5]
    n_heads_values = [2, 4, 6, 8]
    dropout_values = [0.3, 0.4, 0.5, 0.6, 0.7]

    best_params = None
    best_val_acc = 0.0

    if optimize_params:
        print("Optimizing parameters using grid search...")
        for lr, proj_dim, n_blocks, n_heads, dropout in product(
            lr_values, proj_dim_values, n_blocks_values, n_heads_values, dropout_values
        ):
            # Ensure proj_dim is divisible by n_heads
            if proj_dim % n_heads != 0:
                continue

            model_instance = model_class(
                num_classes=num_classes,
                proj_dim=proj_dim,
                n_blocks=n_blocks,
                n_heads=n_heads,
                dropout=dropout
            ).to(device)

            optimizer = optim.Adam(model_instance.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            val_acc = 0.0
            for epoch in range(tuner_epochs):
                model_instance.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.long().to(device)
                    optimizer.zero_grad()
                    outputs = model_instance(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                model_instance.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.long().to(device)
                        outputs = model_instance(inputs)
                        preds = torch.argmax(outputs, dim=1)
                        val_preds.extend(preds.cpu().numpy())
                        val_labels.extend(labels.cpu().numpy())

                val_acc = accuracy_score(val_labels, val_preds)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {
                    'lr': lr,
                    'proj_dim': proj_dim,
                    'n_blocks': n_blocks,
                    'n_heads': n_heads,
                    'dropout': dropout
                }
    else:
        print("Skipping parameter optimization as `optimize_params=False`.")
        best_params = {
            'lr': 1e-3,
            'proj_dim': 64,
            'n_blocks': 4,
            'n_heads': 8,
            'dropout': 0.5
        }

    if not run_final_train:
        print("Skipping final training as `run_final_train=False`.")
        return

    # --- Final Training ---
    print("\nRetraining best model from scratch...")
    best_model = model_class(
        num_classes=num_classes,
        proj_dim=best_params['proj_dim'], # type: ignore
        n_blocks=best_params['n_blocks'], # type: ignore
        n_heads=best_params['n_heads'], # type: ignore
        dropout=best_params['dropout'] # type: ignore
    ).to(device)

    optimizer = optim.Adam(best_model.parameters(), lr=best_params['lr']) # type: ignore
    criterion = nn.CrossEntropyLoss()

    initialize_metric_logs(final_metrics_dir)

    for epoch in range(tuner_epochs):
        best_model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        best_model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.long().to(device)
                outputs = best_model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)

        log_epoch_metrics(final_metrics_dir, epoch, avg_train_loss, avg_val_loss, train_acc, val_acc)

    save_confusion_matrix(final_metrics_dir, val_labels, val_preds) # type: ignore
    print(f"Final Validation Accuracy: {val_acc * 100:.2f}%") # type: ignore

    final_model_path = base_ckpt_dir / "final_best_tuned_vit_model.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'params': best_params
    }, final_model_path)
    print(f"Fully trained model saved at: {final_model_path}")

    return run_id  # Return the UUID for user reference
