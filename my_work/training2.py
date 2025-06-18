import torch
import torch.nn as nn
from monai.data import PersistentDataset, DataLoader
from monai.transforms import Compose
# CHANGE: Import classification metrics from MONAI
from monai.metrics import ROCAUCMetric

import os
import time
import datetime
import calendar
import numpy as np
from typing import List, Dict

from src.my_work.graph3 import ViT 
from src.helpers.utils import get_date_time, save_results

def get_multiclass_label(seg_mask: torch.Tensor) -> torch.Tensor:
    """
    Generates a single multi-class label for an entire 3D segmentation mask.
    The label corresponds to the tumor class with the largest volume.
    If no tumor is present, the label is 0.

    BraTS Labels: 1 (NCR/NET), 2 (ED), 3 (ET). We use these directly.

    Args:
        seg_mask (torch.Tensor): A batch of segmentation masks with shape (B, 1, D, H, W).

    Returns:
        torch.Tensor: A tensor of labels with shape (B,).
    """
    labels = []
    # Process each item in the batch
    for i in range(seg_mask.shape[0]):
        mask = seg_mask[i]
        if mask.sum() == 0:
            labels.append(0)  # Class 0: Background/No Tumor
            continue

        # Count voxels for each class (1, 2, 3)
        # Note: bincount requires a 1D tensor
        counts = torch.bincount(mask.flatten().long())
        
        # Find the label of the most prominent tumor class
        # We ignore the count for class 0 (background)
        if len(counts) > 1:
            # Get the index (which is the class label) of the max value in counts[1:]
            largest_tumor_class = counts[1:].argmax() + 1
            labels.append(largest_tumor_class.item())
        else:
            # This case happens if the mask only contains 0s, but was caught by the sum() check.
            # It's here for robustness.
            labels.append(0)
            
    return torch.tensor(labels, device=seg_mask.device).long()


def train_vit_classifier(
    model: ViT,
    train_files: List[Dict[str, str]],
    val_files: List[Dict[str, str]],
    train_transform: Compose,
    val_transform: Compose,
    epochs: int,
    device: str,
    output_paths: Dict[str, str],
    val_interval: int = 1,
    early_stopping: int = 10,
    num_workers: int = 4,
    ministep: int = 12,
    batch_size: int = 1,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    run_id: str = '',
    write_to_file: bool = True,
    verbose: bool = False
):
    """
    Training function adapted for 4-CLASS supervised classification with a ViT.
    The label for each image is determined by the largest tumor sub-region.
    """
    # 1. --- Initialization ---
    torch_device = torch.device(device)
    model = model.to(torch_device)
    saved_path = output_paths['saved_path']
    reports_path = output_paths['reports_path']
    logs_path = output_paths['logs_path']

    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)

    run_id = run_id or f"{model.name.upper()}_4CLASS_{calendar.timegm(time.gmtime())}"
    cache_dir = os.path.join(output_paths['saved_path'], f'persistent_cache_{run_id}')
    os.makedirs(cache_dir, exist_ok=True)
    
    if len(train_files) < 50 or len(val_files) < 50:
        ministep = 2

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    # --- METRIC CONFIGURATION FOR MULTI-CLASS ---
    # We use "macro" averaging to calculate metrics independently for each class and then average them.
    # This treats all classes equally, regardless of their prevalence.
    auc_metric = ROCAUCMetric(average="macro")

    best_metric, best_metric_epoch = -1, -1
    epoch_losses = {"train": [], "eval": []}
    epoch_metrics = {"accuracy": [], "auc": []}
    epoch_times = []

    log_file = os.path.join(logs_path, f'training_classifier_{run_id}.log')
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f'[{get_date_time()}] Training started for 4-class classification. RUN_ID: {run_id}\n')
        log.flush()

    total_start_time = time.time()

    # 2. --- Training Loop ---
    for epoch in range(epochs):
        epoch_start_time = time.time()
        print("-" * 60)
        print(f"Epoch {epoch + 1}/{epochs} | Run ID: {run_id}")
        
        # --- Training Phase ---
        model.train()
        epoch_loss_train = 0
        step_train = 0
        ministeps_train = np.linspace(0, len(train_files), ministep, dtype=int)

        for i in range(len(ministeps_train) - 1):
            train_ds = PersistentDataset(
                data=train_files[ministeps_train[i]:ministeps_train[i+1]],
                transform=train_transform,
                cache_dir=os.path.join(cache_dir, f'train_part_{i}')
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            for batch_data in train_loader:
                step_train += 1
                inputs = batch_data['image'].to(torch_device)
                seg_labels = batch_data['label'].to(torch_device)

                # --- CHANGE: Generate a multi-class label (0, 1, 2, or 3) ---
                labels = get_multiclass_label(seg_labels)

                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs, _ = model(inputs)
                    loss = loss_function(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss_train += loss.item()

        lr_scheduler.step()
        avg_train_loss = epoch_loss_train / step_train
        epoch_losses["train"].append(avg_train_loss)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_loss_eval = 0
            step_eval = 0
            ministeps_eval = np.linspace(0, len(val_files), ministep, dtype=int)
            
            with torch.no_grad():
                for i in range(len(ministeps_eval) - 1):
                    val_ds = PersistentDataset(
                        data=val_files[ministeps_eval[i]:ministeps_eval[i+1]],
                        transform=val_transform,
                        cache_dir=os.path.join(cache_dir, f'val_part_{i}')
                    )
                    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    
                    for val_data in val_loader:
                        step_eval += 1
                        val_inputs = val_data['image'].to(torch_device)
                        val_seg_labels = val_data['label'].to(torch_device)
                        
                        # --- CHANGE: Generate multi-class label for validation data ---
                        val_labels = get_multiclass_label(val_seg_labels)
                        
                        with torch.cuda.amp.autocast():
                             val_outputs, _ = model(val_inputs)
                             val_loss = loss_function(val_outputs, val_labels)
                        
                        epoch_loss_eval += val_loss.item()
                        
                        # --- CHANGE: Apply softmax to logits before passing to metrics ---
                        # This is required for multi-class AUC calculation.
                        val_outputs_probs = torch.softmax(val_outputs, dim=1)
                        
                        # acc_metric(y_pred=val_outputs_probs, y=val_labels)
                        auc_metric(y_pred=val_outputs_probs, y=val_labels)

            # Aggregate and reset metrics
            avg_eval_loss = epoch_loss_eval / step_eval
            # avg_acc = acc_metric.aggregate().item()
            avg_auc = auc_metric.aggregate().item()
            # acc_metric.reset()
            auc_metric.reset()
            
            epoch_losses["eval"].append(avg_eval_loss)
            # epoch_metrics["accuracy"].append(avg_acc)
            epoch_metrics["auc"].append(avg_auc)

            print(f"Epoch {epoch + 1} Average Validation Loss: {avg_eval_loss:.4f}")
            # print(f"  Metrics -> Macro Accuracy: {avg_acc:.4f}, Macro AUC: {avg_auc:.4f}")

            # Save model based on the primary metric (e.g., Macro AUC)
            if avg_auc > best_metric:
                best_metric = avg_auc
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(saved_path, f'{run_id}_best_model.pth'))
                print(f"  ----> Saved new best model with Macro AUC: {best_metric:.4f} at epoch {best_metric_epoch}")
            
            log_msg = (
                f"[{get_date_time()}] EPOCH {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, ",
                # f"Macro Accuracy: {avg_acc:.4f}, 
                f"Macro AUC: {avg_auc:.4f}\n"
            )
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(log_msg)

            if write_to_file:
                metrics_data = {
                    'id': run_id, 'epoch': epoch + 1, 'model': model.name,
                    'train_loss': avg_train_loss, 'eval_loss': avg_eval_loss,
                    # 'metric_accuracy': avg_acc, 
                    'metric_auc': avg_auc,
                    'exec_time_sec': time.time() - epoch_start_time, 'datetime': get_date_time()
                }
                save_results(
                    file=os.path.join(reports_path, f'{model.name.upper()}_4class_report.csv'),
                    metrics=metrics_data
                )

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} Time: {datetime.timedelta(seconds=int(epoch_time))}")

        if epoch + 1 - best_metric_epoch >= early_stopping:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}. No improvement in the last {early_stopping} epochs.")
            break

    total_time = time.time() - total_start_time
    print("\n" + "="*60)
    print(f"Training complete in {datetime.timedelta(seconds=int(total_time))}")
    print(f"Best Macro AUC: {best_metric:.4f} achieved at epoch: {best_metric_epoch}")
    
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f'[{get_date_time()}] Training finished. RUN_ID: {run_id}\n')
        
    return {
        "train_loss": epoch_losses["train"],
        "eval_loss": epoch_losses["eval"],
        "eval_metrics": epoch_metrics,
        "epoch_times": epoch_times,
    }