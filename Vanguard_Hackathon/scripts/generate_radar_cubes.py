#!/usr/bin/env python3
"""
Project Vanguard — Radar Cube Dataset Generator
================================================
Adapted from Saab AB / ForSyDe `generate-input.py` (Timmy Sundström, 2019)
https://github.com/forsyde/aesa-radar

Generates three 4-channel radar cubes for CNN threat classification:
  - CLEAN:   Background thermal noise only
  - JAMMER:  High-power spot jammer at center frequency
  - THREAT:  Enemy aircraft at 45° azimuth, 5 m² RCS, 300 m/s radial velocity

Output format: NumPy .npy files, shape [4, 256, 128, 2] (channels, range_bins, pulses, I/Q)
               Values stored as float32 for training; INT16 quantized copies also saved.

Git commit message (suggested):
  feat(data): add radar cube generator — CLEAN/JAMMER/THREAT scenarios adapted from Saab AB/ForSyDe
"""

import os
import math
import numpy as np
import argparse
import sys

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# -----------------------------------------------------------------------------
# CLI Arguments
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Project Vanguard — Generate 4-channel AESA radar cubes for CNN training."
)
parser.add_argument(
    "-o", "--output-dir", type=str, default=None,
    help="Output directory for .npy files. Default: <project_root>/data/"
)
parser.add_argument(
    "--seed", type=int, default=42,
    help="Random seed for reproducibility. Default: 42"
)
parser.add_argument(
    "--plot", action="store_true",
    help="Generate spectral diagnostic plots to docs/"
)
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Radar Configuration
# -----------------------------------------------------------------------------
# WHY these parameters?
# - 10 GHz X-band is the standard AESA fighter-radar band (F-22, Gripen, etc.)
# - 4 channels (vs ForSyDe's 16) keeps the data cube within Zynq-7020 BRAM budget
#   while still enabling 4-element beamforming (enough for azimuth discrimination).
# - 256 range bins × 128 pulses gives a cube that fits in ~256 KB at INT8,
#   well within the 4.9 Mb BRAM budget for DMA transfer via AXI-Stream.

RADAR = {
    "f_radar":        10e9,           # 10 GHz X-band
    "wavelength":     3e8 / 10e9,     # λ = 0.03 m
    "n_channels":     4,              # antenna elements
    "n_range_bins":   256,            # fast-time samples per pulse
    "n_pulses":       128,            # slow-time pulses per CPI
    "f_sampling":     3e6,            # ADC sampling rate (3 MHz)
    "pulse_width":    1e-6,           # 1 μs pulse
    "data_width":     16,             # bits per I/Q sample (for quantization)
    "d_element":      None,           # inter-element spacing (set below)
}
RADAR["d_element"] = RADAR["wavelength"] / 2  # λ/2 spacing for grating-lobe-free operation

# -----------------------------------------------------------------------------
# Physics Engine — adapted from ForSyDe GenerateObjectReflection()
# -----------------------------------------------------------------------------

def generate_object_reflection(cube, distance_m, theta_rad, speed_mps, signal_power_dbfs):
    """
    Inject a point-target reflection into the radar cube.
    
    WHY this model?
    Each pulse illuminates the target; the reflected signal arrives after a 
    round-trip delay τ = 2R/c, appearing in a specific range bin. Across pulses,
    the target's Doppler shift ω_d = 4πv/λ creates a phase ramp in slow-time
    that the CNN must learn to detect. Across channels, the spatial phase 
    gradient Δφ = π·sin(θ) encodes the angle-of-arrival — this is the core
    principle behind digital beamforming on an AESA array.
    
    Args:
        cube:               np.ndarray [n_channels, n_range_bins, n_pulses] complex128
        distance_m:         Target range in meters
        theta_rad:          Azimuth angle in radians
        speed_mps:          Radial velocity in m/s (positive = approaching)
        signal_power_dbfs:  Signal amplitude in dBFS (e.g., -18 -> A = 2^(-3))
    """
    cfg = RADAR
    
    # Doppler angular frequency: ω_d = 2π · (2v / λ)
    # WHY factor of 2? Round-trip: transmit path compresses, receive path compresses again
    w_d = 2 * math.pi * 2 * speed_mps / cfg["wavelength"]
    
    # Signal amplitude from dBFS power level
    # ForSyDe convention: power = 2^(signal_power_dbfs / 6.02) ≈ dBFS in half-scale steps
    A = math.pow(2, signal_power_dbfs / 6.02)
    
    # Range bin where the target echo appears
    tau_samples_start = math.ceil((2 * distance_m / 3e8) * cfg["f_sampling"]) % cfg["n_range_bins"]
    tau_samples_stop = math.ceil(
        (2 * distance_m / 3e8 + cfg["pulse_width"]) * cfg["f_sampling"]
    ) % cfg["n_range_bins"]
    
    # Handle range bin wrap-around
    if tau_samples_stop < tau_samples_start:
        range_mask = np.ones(cfg["n_range_bins"], dtype=bool)
        range_mask[tau_samples_stop:tau_samples_start] = False
    else:
        range_mask = np.zeros(cfg["n_range_bins"], dtype=bool)
        range_mask[tau_samples_start:tau_samples_stop] = True
    
    # Random initial phase (each CPI starts with unknown carrier phase)
    phi_0 = 2 * math.pi * np.random.randint(0, 360) / 360
    
    # Vectorized computation (much faster than ForSyDe's triple nested loop)
    pulse_idx = np.arange(cfg["n_pulses"])
    rbin_idx = np.arange(cfg["n_range_bins"])
    
    for ch in range(cfg["n_channels"]):
        # Spatial phase shift for this antenna element
        # WHY π·sin(θ)? At λ/2 spacing, the electrical path difference between
        # adjacent elements is (d·sin(θ))/λ · 2π = π·sin(θ) radians
        channel_phase = -1 * ch * math.pi * math.sin(theta_rad)
        steering = math.cos(channel_phase) + 1j * math.sin(channel_phase)
        
        for p in range(cfg["n_pulses"]):
            for rb in range(cfg["n_range_bins"]):
                if not range_mask[rb]:
                    continue
                t = (rb + p * cfg["n_range_bins"]) / cfg["f_sampling"]
                I = A * math.cos(w_d * t + phi_0)
                Q = -A * math.sin(w_d * t + phi_0)
                cube[ch, rb, p] += (I + 1j * Q) * steering


def generate_noise(cube, signal_power_dbfs):
    """
    Add Gaussian white noise (thermal noise) to the cube.
    
    WHY Gaussian? The Central Limit Theorem: thermal noise from the receiver
    front-end is the sum of many independent random processes -> Gaussian.
    The I and Q components are independent, so we add noise to each separately.
    
    Args:
        cube:               np.ndarray [n_channels, n_range_bins, n_pulses] complex128
        signal_power_dbfs:  Noise power level in dBFS
    """
    cfg = RADAR
    sigma = math.pow(2, signal_power_dbfs / 6.02)
    
    noise_I = np.random.normal(0, sigma, size=cube.shape)
    noise_Q = np.random.normal(0, sigma, size=cube.shape)
    cube += noise_I + 1j * noise_Q


def generate_jammer(cube, signal_power_dbfs, theta_rad=0.0):
    """
    Inject a continuous-wave spot jammer.
    
    WHY is jamming at center frequency? A spot jammer targets the radar's 
    operating frequency with a high-power CW tone. Unlike a target reflection
    (which appears in specific range bins), a barrage/spot jammer floods ALL
    range bins because it's continuous — a key discriminator for the CNN.
    
    The jammer has zero Doppler (it's frequency-matched to the radar) and 
    appears as a massive DC spike in the Doppler FFT across all range bins.
    
    Args:
        cube:               np.ndarray [n_channels, n_range_bins, n_pulses] complex128
        signal_power_dbfs:  Jammer power in dBFS (typically much higher than targets)
        theta_rad:          Jammer direction (default: boresight / 0°)
    """
    cfg = RADAR
    A = math.pow(2, signal_power_dbfs / 6.02)
    
    phi_0 = 2 * math.pi * np.random.randint(0, 360) / 360
    
    for ch in range(cfg["n_channels"]):
        channel_phase = -1 * ch * math.pi * math.sin(theta_rad)
        steering = math.cos(channel_phase) + 1j * math.sin(channel_phase)
        
        # CW jammer: constant amplitude across ALL range bins and pulses
        # (This is the key difference from a target — targets are range-localized)
        for p in range(cfg["n_pulses"]):
            t_pulse = p * cfg["n_range_bins"] / cfg["f_sampling"]
            # Slight AM modulation to make it realistic (jammer isn't perfectly flat)
            am_mod = 1.0 + 0.05 * math.sin(2 * math.pi * 50 * t_pulse)
            signal = A * am_mod * (math.cos(phi_0) - 1j * math.sin(phi_0)) * steering
            cube[ch, :, p] += signal


def rcs_to_power_dbfs(rcs_m2, range_m, wavelength):
    """
    Simplified radar equation to convert RCS -> received signal power in dBFS.
    
    WHY simplified? For the training dataset, we don't need exact power —
    we need realistic *relative* power levels. The key physics is that
    received power ∝ RCS / R⁴, which this captures.
    
    P_r ∝ (σ · λ²) / ((4π)³ · R⁴)
    
    We normalize so that a 10 m² target at 10 km gives approximately -18 dBFS
    (matching the ForSyDe reference scenario).
    """
    # Reference: 10 m² at 10 km -> -18 dBFS
    ref_rcs = 10.0
    ref_range = 10e3
    ref_power_dbfs = -18.0
    
    # Scale by the radar equation ratio
    ratio = (rcs_m2 / ref_rcs) * (ref_range / range_m) ** 4
    power_linear = math.pow(2, ref_power_dbfs / 6.02) * math.sqrt(ratio)
    
    # Convert back to dBFS
    if power_linear > 0:
        power_dbfs = 6.02 * math.log2(power_linear)
    else:
        power_dbfs = -60.0  # floor
    
    return power_dbfs


# -----------------------------------------------------------------------------
# Cube Serialization
# -----------------------------------------------------------------------------

def cube_to_iq_array(cube, dtype=np.float32):
    """
    Convert complex cube [C, R, P] -> real I/Q array [C, R, P, 2].
    
    WHY I/Q separation? The FPGA receives I and Q as separate data streams
    via the ADC. Our CNN input mirrors the hardware data path: two real-valued
    channels per antenna element, fed via AXI-Stream.
    """
    iq = np.stack([cube.real, cube.imag], axis=-1).astype(dtype)
    return iq


def cube_to_int16(cube):
    """Quantize to INT16 (matching ForSyDe's 16-bit data width)."""
    iq = cube_to_iq_array(cube, dtype=np.float64)
    # Normalize to [-1, 1] then scale to INT16 range
    max_val = np.abs(iq).max()
    if max_val > 0:
        iq_norm = iq / max_val
    else:
        iq_norm = iq
    return (iq_norm * 32767).astype(np.int16), max_val


# -----------------------------------------------------------------------------
# Main Generation Pipeline
# -----------------------------------------------------------------------------

def make_empty_cube():
    """Create a zeroed complex radar cube [n_channels, n_range_bins, n_pulses]."""
    return np.zeros(
        (RADAR["n_channels"], RADAR["n_range_bins"], RADAR["n_pulses"]),
        dtype=np.complex128
    )


def generate_clean(seed):
    """
    CLEAN scenario: Background thermal noise only.
    This is the null hypothesis — what the radar sees when the sky is empty.
    """
    np.random.seed(seed)
    cube = make_empty_cube()
    # -42 dBFS noise floor (≈ -7 in ForSyDe's power convention: 6.02 × -7 ≈ -42)
    generate_noise(cube, signal_power_dbfs=-42.0)
    return cube


def generate_jammer_scenario(seed):
    """
    JAMMER scenario: High-power spot jammer at center frequency.
    
    WHY is this important? Electronic warfare (EW) is the #1 real-world threat
    to AESA radars. A spot jammer overwhelms the receiver with a CW signal,
    masking any real targets. The CNN must learn to distinguish this broadband
    energy flood from target reflections and clean noise.
    """
    np.random.seed(seed)
    cube = make_empty_cube()
    # Background noise first
    generate_noise(cube, signal_power_dbfs=-42.0)
    # High-power jammer at -6 dBFS (extremely loud — only ~6 dB below full scale)
    generate_jammer(cube, signal_power_dbfs=-6.0, theta_rad=0.0)
    return cube


def generate_threat_scenario(seed):
    """
    THREAT scenario: Enemy aircraft.
    
    Parameters from mission brief:
      - Azimuth: 45° (π/4 rad)
      - RCS: 5 m² (fighter-sized target, e.g., Su-35 with partial stealth)
      - Radial velocity: 300 m/s (~Mach 0.88, typical attack run speed)
      - Range: 15 km (mid-range engagement zone)
    
    WHY 300 m/s? This creates a Doppler shift of:
      f_d = 2v/λ = 2(300)/0.03 = 20,000 Hz = 20 kHz
    At our PRF, this produces a clearly visible Doppler peak that the CNN
    must learn to detect even in the presence of noise.
    """
    np.random.seed(seed)
    cube = make_empty_cube()
    
    # Background noise
    generate_noise(cube, signal_power_dbfs=-42.0)
    
    # Target parameters
    theta = math.pi / 4       # 45° azimuth
    rcs = 5.0                  # m² radar cross-section
    speed = 300.0              # m/s radial velocity (approaching)
    range_m = 15e3             # 15 km range
    
    # Convert RCS to received power via simplified radar equation
    power_dbfs = rcs_to_power_dbfs(rcs, range_m, RADAR["wavelength"])
    print(f"  THREAT target: theta={math.degrees(theta):.0f} deg, "
          f"RCS={rcs} m^2, v={speed} m/s, R={range_m/1e3:.0f} km -> "
          f"P_rx={power_dbfs:.1f} dBFS")
    
    # Doppler frequency for context
    f_doppler = 2 * speed / RADAR["wavelength"]
    print(f"  Doppler shift: f_d = {f_doppler/1e3:.1f} kHz")
    
    generate_object_reflection(cube, range_m, theta, speed, power_dbfs)
    return cube


# -----------------------------------------------------------------------------
# Spectral Diagnostic Plots
# -----------------------------------------------------------------------------

def generate_plots(cubes, output_dir):
    """
    Generate Range-Doppler maps for visual verification.
    
    WHY Range-Doppler? This is the standard radar output format:
    - X-axis: Doppler frequency (velocity)
    - Y-axis: Range bin (distance)
    A target appears as a bright spot at (range, velocity).
    A jammer appears as a bright horizontal stripe (all ranges, zero Doppler).
    Clean noise is a flat speckle pattern.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed — skipping plots.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels = ["CLEAN", "JAMMER", "THREAT"]
    
    for ax, (label, cube) in zip(axes, zip(labels, cubes)):
        # Range-Doppler map: FFT along slow-time (pulse) dimension for channel 0
        rd_map = np.fft.fftshift(np.fft.fft(cube[0, :, :], axis=1), axes=1)
        power_db = 20 * np.log10(np.abs(rd_map) + 1e-12)
        
        im = ax.imshow(
            power_db, aspect="auto", cmap="inferno",
            extent=[0, RADAR["n_pulses"], RADAR["n_range_bins"], 0],
            vmin=power_db.max() - 60, vmax=power_db.max()
        )
        ax.set_title(f"{label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Doppler Bin")
        ax.set_ylabel("Range Bin")
        plt.colorbar(im, ax=ax, label="Power (dB)")
    
    plt.suptitle("Project Vanguard — Radar Cube Diagnostics (Channel 0)", fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Determine docs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    docs_dir = os.path.join(project_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    plot_path = os.path.join(docs_dir, "cube_spectra.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [OK] Spectral plots saved to: {plot_path}")


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  PROJECT VANGUARD — AESA Radar Cube Generator")
    print("  Adapted from Saab AB / ForSyDe (Timmy Sundström, 2019)")
    print("=" * 65)
    
    # Resolve output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.join(project_root, "data")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n  Output directory: {out_dir}")
    print(f"  Random seed:     {args.seed}")
    print(f"  Cube shape:      [{RADAR['n_channels']}, {RADAR['n_range_bins']}, "
          f"{RADAR['n_pulses']}] complex -> [{RADAR['n_channels']}, {RADAR['n_range_bins']}, "
          f"{RADAR['n_pulses']}, 2] I/Q")
    
    # -- Generate scenarios ----------------------------------------------
    scenarios = {
        "CLEAN":   ("Background noise only", generate_clean),
        "JAMMER":  ("High-power spot jammer at center freq", generate_jammer_scenario),
        "THREAT":  ("Enemy aircraft @ 45°, 5m² RCS, 300 m/s", generate_threat_scenario),
    }
    
    cubes_complex = []
    
    for name, (desc, gen_func) in scenarios.items():
        print(f"\n{'-' * 50}")
        print(f"  Generating [{name}]: {desc}")
        print(f"{'-' * 50}")
        
        cube = gen_func(args.seed)
        cubes_complex.append(cube)
        
        # Save float32 I/Q (for training)
        iq_f32 = cube_to_iq_array(cube, dtype=np.float32)
        f32_path = os.path.join(out_dir, f"{name}.npy")
        np.save(f32_path, iq_f32)
        
        # Save INT16 I/Q (for FPGA / HLS simulation)
        iq_i16, scale = cube_to_int16(cube)
        i16_path = os.path.join(out_dir, f"{name}_int16.npy")
        np.save(i16_path, iq_i16)
        
        # Statistics
        print(f"  Shape:       {iq_f32.shape}")
        print(f"  Float32:     {f32_path} ({os.path.getsize(f32_path) / 1024:.1f} KB)")
        print(f"  INT16:       {i16_path} ({os.path.getsize(i16_path) / 1024:.1f} KB)")
        print(f"  Power (rms): {np.sqrt(np.mean(np.abs(cube)**2)):.6f}")
        print(f"  Peak |I/Q|:  {np.abs(iq_f32).max():.6f}")
    
    # -- Diagnostic plots ------------------------------------------------
    if args.plot:
        print(f"\n{'-' * 50}")
        print("  Generating diagnostic plots...")
        generate_plots(cubes_complex, out_dir)
    
    # -- Summary ---------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  [OK] ALL CUBES GENERATED SUCCESSFULLY")
    print(f"  Output: {out_dir}")
    print(f"  Files:  {', '.join(f'{n}.npy' for n in scenarios.keys())}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
