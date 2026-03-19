#!/usr/bin/env python3
"""
Project Vanguard -- Train VanguardNet (Wider 1D CNN)
=====================================================
Per-channel normalization from training set, class weights,
GPU preloading, 150 epochs, early stopping.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VanguardNet, model_summary

CLASS_NAMES = ['THREAT', 'JAMMER', 'CLEAN']


def print_confusion_matrix(preds, labels):
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    print(f'\n  {"Confusion Matrix":^45}')
    print(f'  {"":>12}', end='')
    for name in CLASS_NAMES:
        print(f' {name:>8}', end='')
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f'  {name:>12}', end='')
        for j in range(n):
            print(f' {cm[i, j]:>8}', end='')
        print()
    print(f'\n  {"Class":<12} {"Precision":>10} {"Recall":>10} {"F1":>10}')
    print(f'  {"-" * 42}')
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f'  {name:<12} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}')
    return cm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--no-qat', action='store_true')
    parser.add_argument('--export-onnx', action='store_true')
    args = parser.parse_args()

    print('=' * 60)
    print('  PROJECT VANGUARD -- VanguardNet Training (Wider 1D CNN)')
    print('=' * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Device:     {device}')
    if device.type == 'cuda':
        print(f'  GPU:        {torch.cuda.get_device_name(0)}')
        print(f'  VRAM:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # --- Load data ---
    print(f'\n  Loading dataset...')
    X = np.load(os.path.join(data_dir, 'X_processed.npy'))  # [3000, 16, 256]
    y = np.load(os.path.join(data_dir, 'y_processed.npy'))  # [3000]
    print(f'  X: {X.shape} {X.dtype} | y: {y.shape}')
    print(f'  Class distribution: {np.bincount(y)}')

    # --- Train/val split ---
    n_val = len(X) // 5
    n_train = len(X) - n_val
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    print(f'  Train: {n_train} | Val: {n_val}')

    # --- Per-channel normalization (from TRAINING SET ONLY) ---
    # X shape: [N, 16, 256]
    # Compute mean/std per channel across samples and range bins
    train_mean = X_train.mean(axis=(0, 2), keepdims=True)  # [1, 16, 1]
    train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8  # [1, 16, 1]

    X_train = ((X_train - train_mean) / train_std).astype(np.float32)
    X_val = ((X_val - train_mean) / train_std).astype(np.float32)

    # Save normalization stats for PYNQ inference
    np.save(os.path.join(data_dir, 'norm_mean.npy'), train_mean)
    np.save(os.path.join(data_dir, 'norm_std.npy'), train_std)
    print(f'  Per-channel normalization applied (train stats saved)')

    # --- Pre-load entire dataset to GPU ---
    X_train_gpu = torch.FloatTensor(X_train).to(device)
    y_train_gpu = torch.LongTensor(y_train).to(device)
    X_val_gpu = torch.FloatTensor(X_val).to(device)
    y_val_gpu = torch.LongTensor(y_val).to(device)
    del X, X_train, X_val  # free CPU memory

    if device.type == 'cuda':
        print(f'  GPU memory after load: {torch.cuda.memory_allocated()/1e6:.0f} MB')

    # --- Model ---
    model = VanguardNet(n_classes=3).to(device)
    model_summary(model)

    # THREAT gets 1.5x weight (it was the confused class)
    class_weights = torch.FloatTensor([1.5, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_model_path = os.path.join(data_dir, 'vanguard_best.pth')
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    print(f'\n  Training for up to {args.epochs} epochs '
          f'(patience={args.patience})...')
    print(f'  {"Epoch":>6} {"Tr Loss":>9} {"Tr Acc":>9} '
          f'{"Va Loss":>9} {"Va Acc":>9}')
    print(f'  {"-" * 46}')

    best_val_acc = 0
    patience_counter = 0
    batch_size = args.batch_size

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        perm = torch.randperm(len(X_train_gpu), device=device)
        train_loss, train_correct, train_total = 0, 0, 0

        for i in range(0, len(X_train_gpu), batch_size):
            idx_batch = perm[i:i + batch_size]
            X_batch = X_train_gpu[idx_batch]
            y_batch = y_train_gpu[idx_batch]

            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            train_correct += (out.argmax(1) == y_batch).sum().item()
            train_total += len(X_batch)

        scheduler.step()

        # --- Val ---
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_gpu)
            val_loss_val = criterion(val_out, y_val_gpu).item()
            val_correct = (val_out.argmax(1) == y_val_gpu).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / len(y_val_gpu)
        train_loss_avg = train_loss / train_total

        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss_val)
        history['val_acc'].append(val_acc)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f'  {epoch:>6} {train_loss_avg:>9.4f} {train_acc:>8.1%} '
                  f'{val_loss_val:>9.4f} {val_acc:>8.1%}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f'\n  Early stopping at epoch {epoch} '
                  f'(patience={args.patience})')
            break

        if epoch == 60 and best_val_acc < 0.90:
            print(f'\n  WARNING: val_acc below 90% after 60 epochs '
                  f'({best_val_acc:.1%})')

    print(f'\n  Best val accuracy: {best_val_acc:.1%}')
    print(f'  Model saved: {best_model_path}')

    # --- Save history ---
    hist_path = os.path.join(data_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f)

    # --- Training curves ---
    try:
        from visualize import plot_training_curves
        plot_training_curves(history)
    except Exception as e:
        print(f'  [WARN] Could not plot curves: {e}')

    # --- Final evaluation ---
    print(f'\n{"=" * 60}')
    print('  Final Evaluation')
    print(f'{"=" * 60}')

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        val_out = model(X_val_gpu)
        val_loss_final = criterion(val_out, y_val_gpu).item()
        preds = val_out.argmax(1).cpu().numpy()
        labels = y_val_gpu.cpu().numpy()

    print(f'\n  Val Loss: {val_loss_final:.4f} | Val Acc: '
          f'{(preds == labels).mean():.1%}')
    print_confusion_matrix(preds, labels)

    # --- ONNX export ---
    if args.export_onnx:
        print(f'\n{"=" * 60}')
        print('  ONNX Export')
        print(f'{"=" * 60}')
        onnx_path = os.path.join(data_dir, 'vanguard_net.onnx')
        export_model = VanguardNet(n_classes=3)
        export_model.load_state_dict(
            torch.load(best_model_path, weights_only=True), strict=False)
        export_model.eval()
        dummy = torch.randn(1, 16, 256)
        try:
            torch.onnx.export(
                export_model, dummy, onnx_path,
                input_names=['radar_features'],
                output_names=['threat_class'],
                opset_version=18,
            )
            print(f'  ONNX saved: {onnx_path} '
                  f'({os.path.getsize(onnx_path) / 1024:.1f} KB)')
        except Exception as e:
            print(f'  [WARN] ONNX export failed: {e}')

    print(f'\n{"=" * 60}')
    print('  [OK] TRAINING COMPLETE')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
