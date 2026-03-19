#!/usr/bin/env python3
"""
Project Vanguard -- Visualization (1D CNN, 16-channel)
=======================================================
Plot doppler_max and doppler_idx per class, training curves.
"""

import os
import sys
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def plot_rd_maps(data_dir=None, save_dir=None):
    """Plot per-class feature maps (doppler_max and doppler_idx)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if data_dir is None:
        data_dir = os.path.join(project_root, 'data')
    if save_dir is None:
        save_dir = os.path.join(project_root, 'docs')
    os.makedirs(save_dir, exist_ok=True)

    X = np.load(os.path.join(data_dir, 'X_processed.npy'))
    y = np.load(os.path.join(data_dir, 'y_processed.npy'))
    print(f'  Loaded X: {X.shape} {X.dtype}')

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    names = {0: 'THREAT', 1: 'JAMMER', 2: 'CLEAN'}
    BEAM = 3  # middle beam

    for col, (lbl, name) in enumerate(names.items()):
        subset = X[y == lbl]
        idx = np.random.choice(len(subset), min(20, len(subset)),
                               replace=False)
        samples = subset[idx]

        # Top row: doppler_max (channel=BEAM)
        dmax_map = samples[:, BEAM, :].mean(axis=0)  # [256]
        axes[0, col].plot(dmax_map, linewidth=1)
        axes[0, col].set_title(f'{name} — doppler_max (beam {BEAM})',
                               fontsize=12, fontweight='bold')
        axes[0, col].set_xlabel('Range Bin')
        axes[0, col].set_ylabel('log1p(magnitude)')
        axes[0, col].grid(True, alpha=0.3)

        # Bottom row: doppler_idx (channel=8+BEAM)
        didx_map = samples[:, 8 + BEAM, :].mean(axis=0)  # [256]
        axes[1, col].plot(didx_map, linewidth=1, color='orange')
        axes[1, col].set_title(f'{name} — doppler_idx (beam {BEAM})',
                               fontsize=12, fontweight='bold')
        axes[1, col].set_xlabel('Range Bin')
        axes[1, col].set_ylabel('Normalized Doppler bin')
        axes[1, col].set_ylim(-0.1, 1.1)
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('Project Vanguard — Feature Maps (16-channel)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, 'rd_maps_processed.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_training_curves(history, save_dir=None):
    """Plot training loss/accuracy curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if save_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(os.path.dirname(script_dir), 'docs')
    os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], label='Train', linewidth=2)
    ax1.plot(epochs, history['val_loss'], label='Val', linewidth=2)
    ax1.set_title('Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], label='Train', linewidth=2)
    ax2.plot(epochs, history['val_acc'], label='Val', linewidth=2)
    ax2.set_title('Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5,
                label='>95% target')

    plt.suptitle('Project Vanguard — Training Curves (Wider 1D CNN)',
                 fontsize=14)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


if __name__ == '__main__':
    np.random.seed(42)
    plot_rd_maps()
