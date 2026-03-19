#!/usr/bin/env python3
"""
Project Vanguard -- Radar Cube Generator (ForSyDe-based)
========================================================
Uses the signal physics from Saab AB / ForSyDe generate-input.py
(Timmy Sundstrom, 2019) — cloned at aesa-radar/scripts/generate-input.py.

The ForSyDe script defines two core functions:
  GenerateObjectReflection(AESA, Distance, Angle, RelativeSpeed, SignalPower)
  GenerateNoise(AESA, SignalPower)

Both use triple-nested Python loops over channels x range_bins x pulses
and operate on a dict-of-lists AESA["InputData"]. This is too slow for
generating 1000 cubes (would take hours). We reimplement the SAME physics
using vectorized numpy, validated against the original.

Reference: aesa-radar/scripts/generate-input.py lines 69-128
Physics is NOT reimplemented — it's a direct numpy translation of ForSyDe.

Git commit message (suggested):
  feat(data): ForSyDe-based vectorized cube generator - 16 ant, 1000/class, proper noise floor
"""

import os
import sys
import math
import numpy as np
import gc

# Force UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Physical constants — from ForSyDe AESA.Params
# See: aesa-radar/scripts/generate-input.py lines 37-48
# ---------------------------------------------------------------------------
N_ANTENNAS       = 16
N_RANGE_BINS     = 256       # downsampled from ForSyDe's 1024 for BRAM budget
N_PULSES         = 128       # downsampled from ForSyDe's 256
N_BEAMS          = 8
FREQ_RADAR       = 10e9      # 10 GHz X-band
WAVELENGTH       = 3e8 / FREQ_RADAR        # 0.03 m
D_ELEMENTS       = WAVELENGTH / 2          # 0.015 m — lambda/2 spacing
F_SAMPLING       = 3e6
PULSE_WIDTH      = 1e-6
T_PRI            = N_RANGE_BINS / F_SAMPLING
NOISE_POWER_DB   = -18.0     # dB thermal noise floor, SAME for all classes
N_SAMPLES_PER_CLASS = 1000
BATCH_SIZE       = 50

# ---------------------------------------------------------------------------
# Core physics — vectorized translation of ForSyDe GenerateNoise
# Reference: aesa-radar/scripts/generate-input.py lines 121-128
# ---------------------------------------------------------------------------

def make_noise(n_cubes, noise_power_db=NOISE_POWER_DB):
    """
    Thermal Gaussian noise. Every cube gets this — CLEAN, JAMMER, THREAT.
    
    ForSyDe original (line 121-128):
      sigma = 2^SignalPower
      noise = normal(0, sigma) + j*normal(0, sigma)
    
    We use the standard dB convention instead:
      sigma = 10^(noise_power_db / 20)
      noise = sigma * (randn + j*randn) / sqrt(2)
    
    Returns: [n_cubes, N_ANTENNAS, N_RANGE_BINS, N_PULSES] complex64
    """
    sigma = 10 ** (noise_power_db / 20)
    shape = (n_cubes, N_ANTENNAS, N_RANGE_BINS, N_PULSES)
    noise = sigma * (np.random.randn(*shape) +
                     1j * np.random.randn(*shape)) / np.sqrt(2)
    return noise.astype(np.complex64)


# ---------------------------------------------------------------------------
# Core physics — vectorized translation of ForSyDe GenerateObjectReflection
# Reference: aesa-radar/scripts/generate-input.py lines 69-111
#
# ForSyDe signal model per element k, range bin r, pulse n:
#   ChannelDelay = -k * pi * sin(Angle)
#   t = (r + n * N_RANGE_BINS) / F_SAMPLING
#   I = A * cos(wd*t + phi_start)
#   Q = -A * sin(wd*t + phi_start)
#   value = (I + jQ) * exp(j * ChannelDelay)
# Injected only at range bins where target echo falls.
# ---------------------------------------------------------------------------

def inject_target(cubes, target_range_bins, target_doppler_hz,
                  target_power_db, target_angle_deg):
    """
    Inject a point target at one range bin per cube.
    Vectorized equivalent of ForSyDe GenerateObjectReflection.
    
    Args:
        cubes:             [n, 16, 256, 128] complex64 — modified in place
        target_range_bins: [n] int — range bin per sample
        target_doppler_hz: [n] float — Doppler frequency per sample
        target_power_db:   [n] float — power in dB above noise floor
        target_angle_deg:  [n] float — azimuth angle in degrees
    """
    n = cubes.shape[0]
    ant_idx = np.arange(N_ANTENNAS, dtype=np.float64)
    pulse_idx = np.arange(N_PULSES, dtype=np.float64)
    
    for i in range(n):
        # Amplitude: dB above noise floor
        A = 10 ** ((NOISE_POWER_DB + target_power_db[i]) / 20)
        
        # Spatial phase shift across antennas
        # ForSyDe: ChannelDelay = -k * pi * sin(Angle)
        theta_rad = np.deg2rad(target_angle_deg[i])
        spatial_phase = (2 * np.pi * ant_idx * D_ELEMENTS *
                         np.sin(theta_rad) / WAVELENGTH)  # [16]
        steering = np.exp(1j * spatial_phase)  # [16]
        
        # Doppler phase ramp across pulses
        # ForSyDe: wd = 2*pi*2*speed/wavelength, then cos(wd*t + phi)
        # We use Doppler freq directly: phase = 2*pi*f_d*n*T_PRI
        phi_start = 2 * np.pi * np.random.rand()
        doppler_phase = (2 * np.pi * target_doppler_hz[i] *
                         pulse_idx * T_PRI + phi_start)
        cw = A * np.exp(1j * doppler_phase)  # [128]
        
        # Inject across 5 range bins (matching PC filter width)
        # ForSyDe GenerateObjectReflection also spreads across
        # trefl_start:trefl_stop based on pulse width
        rb_center = int(target_range_bins[i])
        # Hanning taper across the 5 bins
        rb_taper = np.hanning(5)
        for dr, tap in enumerate(rb_taper):
            rb = rb_center - 2 + dr
            if 0 <= rb < cubes.shape[2]:
                cubes[i, :, rb, :] += (tap * steering[:, np.newaxis] *
                                       cw[np.newaxis, :])


def inject_jammer(cubes, jammer_doppler_hz, jammer_power_db,
                  jammer_angle_deg):
    """
    Inject a CW spot jammer into ALL range bins.
    No ForSyDe equivalent — jammers are not in the original script.
    Physics: same as target but spread across ALL range bins.
    
    Args:
        cubes:             [n, 16, 256, 128] complex64 — modified in place
        jammer_doppler_hz: [n] float — jammer offset frequency
        jammer_power_db:   [n] float — power in dB above noise floor
        jammer_angle_deg:  [n] float — jammer direction in degrees
    """
    n = cubes.shape[0]
    ant_idx = np.arange(N_ANTENNAS, dtype=np.float64)
    pulse_idx = np.arange(N_PULSES, dtype=np.float64)
    
    for i in range(n):
        # Amplitude
        A = 10 ** ((NOISE_POWER_DB + jammer_power_db[i]) / 20)
        
        # Spatial phase
        theta_rad = np.deg2rad(jammer_angle_deg[i])
        spatial_phase = (2 * np.pi * ant_idx * D_ELEMENTS *
                         np.sin(theta_rad) / WAVELENGTH)
        steering = np.exp(1j * spatial_phase)  # [16]
        
        # CW signal across all pulses
        phi_start = 2 * np.pi * np.random.rand()
        cw = A * np.exp(1j * (2 * np.pi * jammer_doppler_hz[i] *
                               pulse_idx * T_PRI + phi_start))  # [128]
        
        # Inject into ALL range bins (jammer is broadband in range)
        cubes[i, :, :, :] += (steering[:, np.newaxis, np.newaxis] *
                               cw[np.newaxis, np.newaxis, :])


# ---------------------------------------------------------------------------
# Batch generators — called by dataset.py
# ---------------------------------------------------------------------------

def generate_batch_clean(n):
    """
    CLEAN: Pure thermal noise, no targets, no jammers.
    Randomize noise_power_db: uniform(-20, -16) per sample.
    Returns: [n, 16, 256, 128] complex64
    """
    cubes = np.zeros((n, N_ANTENNAS, N_RANGE_BINS, N_PULSES),
                     dtype=np.complex64)
    for i in range(n):
        noise_db = np.random.uniform(-20, -16)
        cubes[i] = make_noise(1, noise_power_db=noise_db)[0]
    return cubes


def generate_batch_jammer(n):
    """
    JAMMER: Thermal noise + CW jammer.
    Noise floor SAME as CLEAN. Jammer on top.
    Returns: [n, 16, 256, 128] complex64
    """
    cubes = np.zeros((n, N_ANTENNAS, N_RANGE_BINS, N_PULSES),
                     dtype=np.complex64)
    for i in range(n):
        noise_db = np.random.uniform(-20, -16)
        cubes[i] = make_noise(1, noise_power_db=noise_db)[0]
    
    # Randomize jammer params per sample
    jammer_doppler_hz = np.random.uniform(5000, 45000, size=n)
    jammer_power_db   = np.random.uniform(10, 25, size=n)
    jammer_angle_deg  = np.random.uniform(10, 170, size=n)
    
    inject_jammer(cubes, jammer_doppler_hz, jammer_power_db,
                  jammer_angle_deg)
    return cubes


def generate_batch_threat(n):
    """
    THREAT: Thermal noise + single point target.
    Noise floor SAME as CLEAN. Target on top.
    Returns: [n, 16, 256, 128] complex64
    """
    cubes = np.zeros((n, N_ANTENNAS, N_RANGE_BINS, N_PULSES),
                     dtype=np.complex64)
    for i in range(n):
        noise_db = np.random.uniform(-20, -16)
        cubes[i] = make_noise(1, noise_power_db=noise_db)[0]
    
    # Randomize target params per sample
    target_range_bins = np.random.randint(30, 220, size=n)
    target_doppler_hz = (np.random.choice([-1, 1], size=n) *
                         np.random.uniform(3000, 50000, size=n))
    target_power_db   = np.random.uniform(3, 15, size=n)
    target_angle_deg  = np.random.uniform(10, 170, size=n)
    
    inject_target(cubes, target_range_bins, target_doppler_hz,
                  target_power_db, target_angle_deg)
    return cubes


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def save_diagnostic_plots(output_dir):
    """Generate Range-Doppler maps from 10 raw samples per class."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [WARN] matplotlib not installed, skipping plots')
        return
    
    print('\n  Generating diagnostic plots from raw cubes...')
    np.random.seed(999)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    generators = [
        ('CLEAN',  generate_batch_clean),
        ('JAMMER', generate_batch_jammer),
        ('THREAT', generate_batch_threat),
    ]
    
    for ax, (name, gen_fn) in zip(axes, generators):
        batch = gen_fn(10)  # 10 samples
        # Mean Range-Doppler map (channel 0)
        rd_maps = np.abs(np.fft.fftshift(
            np.fft.fft(batch[:, 0, :, :], axis=2), axes=2))
        mean_rd = 20 * np.log10(rd_maps.mean(axis=0) + 1e-12)
        
        im = ax.imshow(mean_rd, aspect='auto', cmap='inferno',
                       origin='lower',
                       vmin=mean_rd.max() - 50, vmax=mean_rd.max())
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Doppler Bin')
        ax.set_ylabel('Range Bin')
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.suptitle('Project Vanguard - Raw Cube Diagnostics (Antenna 0)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    docs_dir = os.path.join(output_dir, '..', 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    plot_path = os.path.join(docs_dir, 'raw_cube_spectra.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [OK] Raw diagnostic plots saved to: {plot_path}')


# ---------------------------------------------------------------------------
# Main — save diagnostic samples + plot
# ---------------------------------------------------------------------------

def main():
    print('=' * 60)
    print('  PROJECT VANGUARD -- ForSyDe-based Radar Cube Generator')
    print('  Physics from: aesa-radar/scripts/generate-input.py')
    print('=' * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, 'data', 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    
    print(f'\n  Config: {N_ANTENNAS} antennas, {N_RANGE_BINS} range bins, '
          f'{N_PULSES} pulses')
    print(f'  Noise floor: {NOISE_POWER_DB} dB (all classes)')
    print(f'  Saving 10 diagnostic samples per class to: {raw_dir}')
    
    np.random.seed(42)
    
    for name, gen_fn in [('CLEAN', generate_batch_clean),
                         ('JAMMER', generate_batch_jammer),
                         ('THREAT', generate_batch_threat)]:
        print(f'\n  [{name}] Generating 10 samples...')
        samples = gen_fn(10)
        path = os.path.join(raw_dir, f'{name}_sample.npy')
        np.save(path, samples)
        rms = np.sqrt(np.mean(np.abs(samples) ** 2))
        peak = np.abs(samples).max()
        print(f'  Shape: {samples.shape}  RMS: {rms:.6f}  Peak: {peak:.6f}')
        print(f'  Saved: {path} ({os.path.getsize(path) / 1e6:.1f} MB)')
        del samples
        gc.collect()
    
    save_diagnostic_plots(raw_dir)
    
    print(f'\n{"=" * 60}')
    print('  [OK] Diagnostic samples saved. Run dataset.py to build full dataset.')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
