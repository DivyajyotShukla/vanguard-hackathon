# Project Vanguard -- Design Document

## System Overview

AESA radar threat classifier for PYNQ-Z2 (Zynq-7020).
Three classes: **THREAT** (aircraft), **JAMMER** (CW), **CLEAN** (noise).
Target: >95% val accuracy, VanguardNet 1D-CNN, INT8 on PL fabric, <1ms inference.

## Data Generation

Signal physics imported from **Saab AB / ForSyDe** repository:
`aesa-radar/scripts/generate-input.py` (Timmy Sundstrom, 2019).

ForSyDe exposes `GenerateObjectReflection()` and `GenerateNoise()` — both use
triple-nested Python loops (too slow for 1000 samples). Our generator
reimplements the **exact same physics** in vectorized numpy, validated against
the original. Source reference comments point to specific line numbers.

### Parameters (ForSyDe AESA.Params)

| Parameter | Value | Origin |
|---|---|---|
| Antennas | 16 | ForSyDe NoDataChannels |
| Range bins | 256 | Downsampled from ForSyDe's 1024 |
| Pulses | 128 | Downsampled from ForSyDe's 256 |
| Frequency | 10 GHz X-band | ForSyDe Fradar |
| Wavelength | 0.03 m | 3e8 / 10e9 |
| Element spacing | 0.015 m | lambda/2 |
| Sampling | 3 MHz | ForSyDe Fsampling |

## Signal Processing Pipeline

Numpy reimplementation of the ForSyDe Haskell processing chain.
Reference: Ungureanu et al. 2019, sections 2.2.1-2.2.5.

### Stage 1: DBF (Digital Beamforming)
- 16 antennas -> 8 beams
- Taylor window (nbar=4, sll=30dB) from ForSyDe AESA.Coefs
- Beam angles: linspace(pi/3, 2*pi/3, 8) — ForSyDe convention
- Phase: (k - 9.5) * 2*pi*d*sin(theta)/lambda — eq (7)

### Stage 2: PC (Pulse Compression)
- 5-tap Hanning FIR matched filter — ForSyDe mkPcCoefs(5)
- Applied along range axis

### Stage 3: DFB (Doppler Filter Bank)
- Hanning window along pulse axis — ForSyDe mkWeightCoefs
- FFT + magnitude -> real-valued from here

### Stage 4: CFAR

> **DEVIATION**: We use Cell-Averaging CFAR (CA-CFAR) instead of ForSyDe's
> geometric+arithmetic mean combination. CA-CFAR is simpler and sufficient
> for 3-class classification. The ForSyDe implementation combines
> `geomMean . arithmMean` which is more complex to port.

Parameters: 32 reference cells (16 each side), 4 guard cells, alpha=2.0.

### Stage 5: Reduce
Max over Doppler axis: [8, 256, 128] -> [8, 256].

> **KNOWN LIMITATION**: This collapses the Doppler (velocity) dimension.
> A DRFM (Digital Radio Frequency Memory) jammer that replays target-like
> signals at the correct range bin would be indistinguishable from a real target.
> **Upgrade path**: Keep full [8, 256, 128] RD map and use 2D CNN.
> This is a deliberate simplification for the hackathon — the 1D CNN
> VanguardNet (6,787 params) can only accept [8, 256].

## CNN Architecture

VanguardNet: 3x Conv1D + GlobalAvgPool + FC. **6,787 params, 6.6 KB @ INT8.**
Estimated latency: ~13 us @ 125 MHz / 220 DSPs. Target: <1000 us.

## HLS Next Steps (Future Session)

1. **hls4ml**: Export VanguardNet ONNX -> HLS C++ via hls4ml
2. **Vivado HLS**: Synthesize, verify timing at 125 MHz
3. **AXI-Stream wrapper**: TLAST/TVALID/TREADY protocol for DMA
4. **Overlay**: Package as .bit + .hwh for PYNQ
5. **Notebook**: Jupyter demo — load cube, DMA transfer, classify, display
