import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
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


def log_trial_result(log_path, trial_index, hyperparams, val_loss):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        'trial_index': trial_index,
        **hyperparams,
        'val_loss': val_loss
    }

    if log_path.suffix == '.json':
        if log_path.exists():
            with open(log_path, 'r') as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        existing_data.append(log_data)
        with open(log_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
    elif log_path.suffix == '.csv':
        df = pd.DataFrame([log_data])
        if log_path.exists():
            df.to_csv(log_path, mode='a', header=False, index=False)
        else:
            df.to_csv(log_path, index=False)


def fine_tune_model_with_search(model_class, train_loader, val_loader, num_classes, tuner_epochs=50):
    base_ckpt_dir = Path("logs/checkpoints")
    hp_ckpt_dir = base_ckpt_dir / "hp_tuning"
    vit_ckpt_dir = base_ckpt_dir / "ViT_tuning"
    log_path = vit_ckpt_dir / "trial_log.csv"
    metrics_dir = Path("logs/metrics/ViT_tuning")

    search_space = {
        'lr': [1e-2, 1e-3, 1e-4],
        'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
    }

    param_combinations = list(product(*search_space.values()))
    random.shuffle(param_combinations)
    selected_trials = param_combinations[:10]

    best_val_loss = float('inf')
    best_model_state = None
    best_params = {}

    for i, params in enumerate(selected_trials):
        hp = dict(zip(search_space.keys(), params))
        print(f"\nTrial {i + 1}: Testing hyperparameters: {hp}")

        model_instance = model_class(
            num_classes=num_classes,
            dropout=hp['dropout']
        ).to(device)

        optimizer = optim.Adam(model_instance.parameters(), lr=hp['lr'])
        criterion = nn.CrossEntropyLoss()

        initialize_metric_logs(metrics_dir)

        for epoch in range(tuner_epochs):
            model_instance.train()
            train_loss = 0.0
            train_preds, train_labels = [], []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.long().to(device)
                optimizer.zero_grad()
                outputs = model_instance(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            avg_train_loss = train_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)

            model_instance.eval()
            val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.long().to(device)
                    outputs = model_instance(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)

            log_epoch_metrics(metrics_dir, epoch, avg_train_loss, avg_val_loss, train_acc, val_acc)
            save_partial_checkpoint(model_instance, optimizer, i, epoch, hp, hp_ckpt_dir)

        print(f"Validation loss: {avg_val_loss:.4f}") # type: ignore
        log_trial_result(log_path, i, hp, avg_val_loss) # type: ignore

        if avg_val_loss < best_val_loss: # type: ignore
            best_val_loss = avg_val_loss # type: ignore
            best_model_state = model_instance.state_dict()
            best_params = hp.copy()
            best_ckpt_path = vit_ckpt_dir / "best_vit_checkpoint.pth"
            save_best_checkpoint(best_model_state, optimizer.state_dict(), best_params, epoch, best_ckpt_path) # type: ignore

    if not best_params:
        print("❌ No best parameters were found. All trials failed.")
        return

    print(f"\nBest trial parameters: {best_params}")
    print(f"Lowest validation loss: {best_val_loss:.4f}")

    print("\n🔍 Evaluating best model on validation set...")
    best_model = model_class(
        num_classes=num_classes,
        dropout=best_params['dropout']
    ).to(device)

    best_model.load_state_dict(best_model_state)
    best_model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    save_confusion_matrix(metrics_dir, all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Final Validation Accuracy: {acc * 100:.2f}%")
