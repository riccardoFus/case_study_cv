import torch
import torch.nn as nn
# --- MODIFICA: Importa PersistentDataset ---
from monai.data import PersistentDataset, DataLoader 
from monai.transforms import Compose

import os, sys
import time
import datetime
import calendar
import numpy as np
import pandas as pd
from typing import List, Dict, Callable, Any
import torch.nn.functional as F

from src.my_work.graph3 import ViTAutoEncoder3D
from src.modules.metrics import (
	calculate_ssim,
	calculate_psnr,
	calculate_ncc,
	calculate_node_dice
)

from src.helpers.utils import get_date_time, save_results, get_class_weights, get_brats_classes


def train_vit_autoencoder(
    model: ViTAutoEncoder3D,
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
    Funzione di training modificata per gestire grandi dataset senza errori di memoria.
    """
    # 1. --- Initialization ---
    torch_device = torch.device(device)
    model = model.to(torch_device)
    saved_path = output_paths['saved_path']
    reports_path = output_paths['reports_path']
    logs_path = output_paths['logs_path']
    
    # Ensure output directories exist
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Logging setup
    run_id = run_id or f"{model.name.upper()}_{calendar.timegm(time.gmtime())}"
    
    # --- MODIFICA 1: Creare una directory per la cache persistente ---
    # PersistentDataset salverà le immagini trasformate su disco qui,
    # invece di tenerle tutte in RAM. Usare il run_id evita conflitti tra esecuzioni.
    cache_dir = os.path.join(output_paths['saved_path'], f'persistent_cache_{run_id}')
    os.makedirs(cache_dir, exist_ok=True)

    # Use smaller ministep for small datasets
    if len(train_files) < 50 or len(val_files) < 50:
        ministep = 2

    # Loss, Optimizer, and Scheduler
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Automatic Mixed Precision (AMP)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    # Metric and Loss Collectors
    best_metric, best_metric_epoch = -1, -1
    epoch_losses = {"train": [], "eval": []}
    epoch_metrics = {"ssim": [], "psnr": [], "ncc": []}
    epoch_times = []

    log_file = os.path.join(logs_path, 'training.log')
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f'[{get_date_time()}] Training started. RUN_ID: {run_id}\n')
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
            # --- MODIFICA 2: Usare PersistentDataset invece di CacheDataset ---
            # Questo evita di caricare l'intero dataset (o una sua parte) in RAM.
            # I dati vengono pre-elaborati una volta e salvati su disco.
            # Ogni "ministep" ha la sua sottocartella di cache.
            train_ds = PersistentDataset(
                data=train_files[ministeps_train[i]:ministeps_train[i+1]],
                transform=train_transform,
                cache_dir=os.path.join(cache_dir, f'train_part_{i}')
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

            for batch_data in train_loader:
                step_train += 1
                step_start_time = time.time()
                # L'input è ora già della dimensione corretta grazie alla pipeline di trasformazione
                inputs = batch_data['image'].to(torch_device)

                # --- MODIFICA 3: Rimuovere il ridimensionamento manuale ---
                # Questa operazione è ora gestita dalla pipeline 'train_transform',
                # quindi le immagini arrivano qui già della dimensione corretta (es. 64x64x64).
                # inputs = F.interpolate(inputs, size=(64,64,64), mode='trilinear', align_corners=False) # <- RIMOSSO

                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    _, reconstruction = model(inputs)
                    loss = loss_function(reconstruction, inputs)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss_train += loss.item()

                if verbose:
                    step_time = time.time() - step_start_time
                    print(f"  Train Step {step_train}, Loss: {loss.item():.4f}, Time: {step_time:.2f}s")

        lr_scheduler.step()
        avg_train_loss = epoch_loss_train / step_train
        epoch_losses["train"].append(avg_train_loss)
        print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_loss_eval = 0
            metric_ssim, metric_psnr, metric_ncc = 0, 0, 0
            step_eval = 0
            ministeps_eval = np.linspace(0, len(val_files), ministep, dtype=int)
            
            with torch.no_grad():
                for i in range(len(ministeps_eval) - 1):
                    # --- MODIFICA 2 (anche per la validazione) ---
                    val_ds = PersistentDataset(
                        data=val_files[ministeps_eval[i]:ministeps_eval[i+1]],
                        transform=val_transform,
                        cache_dir=os.path.join(cache_dir, f'val_part_{i}')
                    )
                    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                    
                    for val_data in val_loader:
                        step_eval += 1
                        val_inputs = val_data['image'].to(torch_device)
                        # Il ridimensionamento manuale non era presente qui, ma è comunque
                        # importante che 'val_transform' includa il ridimensionamento.
                        
                        with torch.cuda.amp.autocast():
                             _, val_reconstruction = model(val_inputs)
                             val_loss = loss_function(val_reconstruction, val_inputs)
                        
                        epoch_loss_eval += val_loss.item()
                        
                        metric_ssim += calculate_ssim(val_reconstruction, val_inputs)
                        metric_psnr += calculate_psnr(val_reconstruction, val_inputs)
                        metric_ncc += calculate_ncc(val_reconstruction, val_inputs)

            # Average losses and metrics over all validation steps
            avg_eval_loss = epoch_loss_eval / step_eval
            epoch_losses["eval"].append(avg_eval_loss)
            
            avg_ssim = metric_ssim / step_eval
            avg_psnr = metric_psnr / step_eval
            avg_ncc = metric_ncc / step_eval
            
            epoch_metrics["ssim"].append(avg_ssim)
            epoch_metrics["psnr"].append(avg_psnr)
            epoch_metrics["ncc"].append(avg_ncc)

            print(f"Epoch {epoch + 1} Average Validation Loss: {avg_eval_loss:.4f}")
            print(f"  Metrics -> SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, NCC: {avg_ncc:.4f}")

            if avg_ssim > best_metric:
                best_metric = avg_ssim
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(saved_path, f'{run_id}_best_model.pth'))
                print(f"  ----> Saved new best model with SSIM: {best_metric:.4f} at epoch {best_metric_epoch}")
            
            # ... (il resto della funzione rimane invariato) ...
            log_msg = (
                f"[{get_date_time()}] EPOCH {epoch+1}/{epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, "
                f"SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.4f}, NCC: {avg_ncc:.4f}\n"
            )
            with open(log_file, 'a', encoding='utf-8') as log:
                log.write(log_msg)
                log.flush()

            if write_to_file:
                metrics_data = {
                    'id': run_id, 'epoch': epoch + 1, 'model': model.name,
                    'train_loss': avg_train_loss, 'eval_loss': avg_eval_loss,
                    'metric_ssim': avg_ssim, 'metric_psnr': avg_psnr, 'metric_ncc': avg_ncc,
                    'exec_time_sec': time.time() - epoch_start_time, 'datetime': get_date_time()
                }
                save_results(
                    file=os.path.join(reports_path, f'{model.name.upper()}_training_report.csv'),
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
    print(f"Best SSIM: {best_metric:.4f} achieved at epoch: {best_metric_epoch}")
    
    with open(log_file, 'a', encoding='utf-8') as log:
        log.write(f'[{get_date_time()}] Training finished. RUN_ID: {run_id}\n')
        log.flush()
        
    return {
        "train_loss": epoch_losses["train"],
        "eval_loss": epoch_losses["eval"],
        "eval_metrics": epoch_metrics,
        "epoch_times": epoch_times,
    }