#!/usr/bin/env python3
"""
Project Vanguard -- Signal Processing Pipeline (Batched)
=========================================================
ForSyDe-equivalent pipeline: DBF → PC → DFB → log → reduce.

Output: [N, 16, 256] float32
  Channels 0-7:  doppler_max — peak log-magnitude per (beam, range bin)
  Channels 8-15: doppler_idx — normalized Doppler bin of peak (0.0 to 1.0)

WHY doppler_idx? A target has a CONSISTENT Doppler bin across all range
bins and beams (coherent CW). Noise has RANDOM Doppler indices. This is
the key feature that separates THREAT from CLEAN after Doppler collapse.
"""

import numpy as np
from scipy.signal import lfilter

# ---------------------------------------------------------------------------
# Constants (matching ForSyDe generate-input.py)
# ---------------------------------------------------------------------------
N_ANTENNAS = 16
N_BEAMS = 8
N_RANGE_BINS = 256
N_PULSES = 128
D_ELEMENTS = 0.5     # half-wavelength spacing
WAVELENGTH = 1.0     # normalized

# Precompute beam steering matrix: [N_BEAMS, N_ANTENNAS] complex128
# ForSyDe uses 8 beams uniformly spaced from -60° to +60°
_beam_angles = np.linspace(-60, 60, N_BEAMS)
_ant_idx = np.arange(N_ANTENNAS, dtype=np.float64)
BEAM_CONSTS = np.zeros((N_BEAMS, N_ANTENNAS), dtype=np.complex128)
for b, angle_deg in enumerate(_beam_angles):
    theta = np.deg2rad(angle_deg)
    phase = 2 * np.pi * _ant_idx * D_ELEMENTS * np.sin(theta) / WAVELENGTH
    BEAM_CONSTS[b] = np.exp(-1j * phase)

# Taylor window for beam sidelobe suppression
_taylor = np.array([1.0, 1.39, 1.64, 1.39, 1.0, 0.58, 0.32, 0.13])
_taylor = _taylor / _taylor.sum()
for b in range(N_BEAMS):
    BEAM_CONSTS[b] *= _taylor[b]

# Pulse compression filter: 5-tap Hanning FIR
PC_COEFS = (np.hanning(5) / 5.0).astype(np.float32)

# DFB: Hanning window for Doppler FFT
DFB_WINDOW = np.hanning(N_PULSES).astype(np.float32)

# CFAR kept for detection/visualization only
CFAR_GUARD = 2
CFAR_TRAIN = 8
CFAR_ALPHA = 4.0


# ---------------------------------------------------------------------------
# Batched pipeline — runs entire batch at once
# ---------------------------------------------------------------------------

def run_pipeline_batch(batch):
    """
    Batched ForSyDe pipeline.

    Input:  [N, 16, 256, 128] complex64
    Output: [N, 16, 256]      float32

    Channels 0-7:  doppler_max — peak log-mag per (beam, range bin)
    Channels 8-15: doppler_idx — normalized peak Doppler bin [0, 1]
    """
    N = batch.shape[0]

    # --- DBF: [N, 16, 256, 128] → [N, 8, 256, 128] ---
    # tensordot: sum over antenna axis (axis=1 of batch, axis=1 of BEAM_CONSTS)
    beams = np.tensordot(
        batch, BEAM_CONSTS.conj().T, axes=([1], [0])
    ).astype(np.complex64)
    # Shape: [N, 256, 128, 8] — need to transpose to [N, 8, 256, 128]
    beams = beams.transpose(0, 3, 1, 2)

    # --- PC: FIR filter along range axis (axis=2) ---
    beams = lfilter(PC_COEFS, [1.0], beams, axis=2).astype(np.complex64)

    # --- DFB: windowed FFT along Doppler axis (axis=3) ---
    windowed = beams * DFB_WINDOW[np.newaxis, np.newaxis, np.newaxis, :]
    spectrum = np.fft.fft(windowed, axis=3)
    rd_map = np.abs(spectrum).astype(np.float32)  # [N, 8, 256, 128]

    # --- Log magnitude ---
    rd_map = np.log1p(rd_map)

    # --- Doppler max + argmax ---
    doppler_max = rd_map.max(axis=3)                         # [N, 8, 256]
    doppler_idx = rd_map.argmax(axis=3).astype(np.float32)   # [N, 8, 256]
    doppler_idx = doppler_idx / N_PULSES                     # normalize to [0, 1]

    # --- Concatenate: [N, 16, 256] ---
    output = np.concatenate([doppler_max, doppler_idx], axis=1)
    return output.astype(np.float32)


def run_pipeline(cube):
    """Single-sample wrapper for inference."""
    return run_pipeline_batch(cube[np.newaxis, :])[0]


# ---------------------------------------------------------------------------
# Legacy per-sample pipeline stages (kept for visualization)
# ---------------------------------------------------------------------------

def dbf(cube_raw):
    """[16, 256, 128] complex → [8, 256, 128] complex."""
    return np.tensordot(BEAM_CONSTS.conj(), cube_raw,
                        axes=([1], [0])).astype(np.complex64)

def pc(cube):
    """[8, 256, 128] complex → [8, 256, 128] complex."""
    return lfilter(PC_COEFS, [1.0], cube, axis=1).astype(np.complex64)

def dfb(cube):
    """[8, 256, 128] complex → [8, 256, 128] float32."""
    windowed = cube * DFB_WINDOW[np.newaxis, np.newaxis, :]
    return np.abs(np.fft.fft(windowed, axis=2)).astype(np.float32)

def cfar(rd_map):
    """Cell-averaging CFAR on [8, 256, 128] float32."""
    out = np.zeros_like(rd_map)
    half = CFAR_GUARD + CFAR_TRAIN
    for b in range(rd_map.shape[0]):
        for r in range(rd_map.shape[1]):
            row = rd_map[b, r, :]
            for d in range(len(row)):
                lo = max(0, d - half)
                hi = min(len(row), d + half + 1)
                guard_lo = max(0, d - CFAR_GUARD)
                guard_hi = min(len(row), d + CFAR_GUARD + 1)
                train_cells = np.concatenate([row[lo:guard_lo], row[guard_hi:hi]])
                if len(train_cells) > 0:
                    threshold = CFAR_ALPHA * train_cells.mean()
                    out[b, r, d] = row[d] / (threshold + 1e-10)
    return out


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Pipeline test: random batch of 4 cubes')
    batch = (np.random.randn(4, N_ANTENNAS, N_RANGE_BINS, N_PULSES) +
             1j * np.random.randn(4, N_ANTENNAS, N_RANGE_BINS, N_PULSES)
             ).astype(np.complex64) * 0.01
    out = run_pipeline_batch(batch)
    print(f'  Input:  {batch.shape} {batch.dtype}')
    print(f'  Output: {out.shape} {out.dtype}')
    print(f'  doppler_max range: [{out[:, :8].min():.3f}, {out[:, :8].max():.3f}]')
    print(f'  doppler_idx range: [{out[:, 8:].min():.3f}, {out[:, 8:].max():.3f}]')
    print('  [OK]')

    # Single sample test
    single = run_pipeline(batch[0])
    print(f'\n  Single: {single.shape} {single.dtype}')
    assert single.shape == (16, 256), f'Expected (16, 256), got {single.shape}'
    print('  [OK] Single pipeline test passed')
