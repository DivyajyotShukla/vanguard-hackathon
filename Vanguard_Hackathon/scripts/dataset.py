#!/usr/bin/env python3
"""
Project Vanguard -- Dataset Builder (Batched Pipeline)
=======================================================
Generates 3000 radar cubes, runs batched pipeline, saves raw output.
Output: [3000, 16, 256] float32 — NO normalization (done in train.py).
"""

import os
import sys
import time
import numpy as np
import gc

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_radar_cubes import (
    generate_batch_clean, generate_batch_jammer, generate_batch_threat,
    N_SAMPLES_PER_CLASS, BATCH_SIZE
)
from pipeline import run_pipeline_batch


def build_dataset():
    print('=' * 60)
    print('  PROJECT VANGUARD -- Dataset Builder (Batched)')
    print('=' * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)

    total = 3 * N_SAMPLES_PER_CLASS
    print(f'\n  Samples per class: {N_SAMPLES_PER_CLASS}')
    print(f'  Batch size:        {BATCH_SIZE}')
    print(f'  Output shape:      [{total}, 16, 256]')

    X_all = []
    y_all = []

    generators = [
        (generate_batch_threat,  0, 'THREAT'),
        (generate_batch_jammer,  1, 'JAMMER'),
        (generate_batch_clean,   2, 'CLEAN'),
    ]

    for gen_fn, label, name in generators:
        print(f'\n  Generating {name} (label={label}): '
              f'{N_SAMPLES_PER_CLASS} samples...')
        class_outputs = []
        t_start = time.time()

        for i in range(0, N_SAMPLES_PER_CLASS, BATCH_SIZE):
            batch_n = min(BATCH_SIZE, N_SAMPLES_PER_CLASS - i)

            # Generate raw cubes: [batch_n, 16, 256, 128] complex64
            raw = gen_fn(batch_n)

            # Batched pipeline: [batch_n, 16, 256] float32
            processed = run_pipeline_batch(raw)
            class_outputs.append(processed)

            del raw
            gc.collect()

            done = min(i + batch_n, N_SAMPLES_PER_CLASS)
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (N_SAMPLES_PER_CLASS - done) / rate if rate > 0 else 0
            print(f'    {done:>4}/{N_SAMPLES_PER_CLASS} '
                  f'({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)')

        X_class = np.concatenate(class_outputs)  # [1000, 16, 256]
        X_all.append(X_class)
        y_all.append(np.full(len(X_class), label, dtype=np.int64))

        elapsed = time.time() - t_start
        print(f'  {name} complete: {len(X_class)} samples in {elapsed:.0f}s')
        del class_outputs
        gc.collect()

    X = np.concatenate(X_all).astype(np.float32)  # [3000, 16, 256]
    y = np.concatenate(y_all)                      # [3000]
    del X_all
    gc.collect()

    print(f'\n  X shape: {X.shape} {X.dtype} ({X.nbytes / 1e6:.1f} MB)')

    # --- Sanity check ---
    print('\n  Per-class statistics (raw, no normalization):')
    for lbl, name in [(0, 'THREAT'), (1, 'JAMMER'), (2, 'CLEAN')]:
        s = X[y == lbl]
        dmax = s[:, :8, :]   # doppler_max channels
        didx = s[:, 8:, :]   # doppler_idx channels
        ptm = dmax.max(axis=(1, 2)) / (dmax.mean(axis=(1, 2)) + 1e-8)
        idx_std = didx.std(axis=2).mean()  # std of doppler idx across range
        print(f'    {name}: dmax_mean={dmax.mean():.4f} dmax_peak={dmax.max():.2f} '
              f'ptm={ptm.mean():.2f} didx_std={idx_std:.4f}')

    # --- Save ---
    x_path = os.path.join(data_dir, 'X_processed.npy')
    y_path = os.path.join(data_dir, 'y_processed.npy')
    np.save(x_path, X)
    np.save(y_path, y)

    print(f'\n  Saved X: {X.shape} → {x_path}')
    print(f'  Saved y: {y.shape} → {y_path}')
    print(f'  Class counts: {np.bincount(y)}')

    print(f'\n{"=" * 60}')
    print('  [OK] Dataset build complete')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    build_dataset()
