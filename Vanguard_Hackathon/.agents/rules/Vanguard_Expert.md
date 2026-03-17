---
trigger: always_on
---

# Team Vanguard: Mission Intelligence & Architectural Rules

## 1. Executive Context
- **Lead Developer**: Divyajyot Shukla (BITS Pilani, 2023A8PS0706P).
- **Core Objective**: Develop a "Crazy-Level" AESA Radar Defense system for the 2026 FPGA Hackathon.
- **Hardware Target**: PYNQ-Z2 (Xilinx Zynq-7020 SoC).
- **Domain**: Edge-AI accelerated Digital Beamforming and Threat Engagement logic.

## 2. Technical Heritage & Expertise
The agent must leverage and respect the developer's existing technical stack:
- **Architecture**: Experience with 32-bit RISC-V processor design and Microprocessor Interfacing.
- **DSP/Analog**: Deep understanding of Sigma-Delta ADCs (CSIR-CEERI) and Digital Decimation Filters.
- **Control Theory**: Expert in Fuzzy Logic Controllers (FLC) and Gaussian/Exponential function implementation on Zynq-7010.
- **Security**: Knowledge of dual-phase shift physical layer security beamforming algorithms.

## 3. Hardware Constraints (Zynq-7020)
- **LUTs**: 53,200 | **DSPs**: 220 | **BRAM**: 4.9 Mb.
- **Timing**: Target a 125MHz system clock ($T = 8.0ns$). 
- **Guideline**: Optimize for DSP48 slice utilization for the 1D-CNN and Beamformer phase-rotations.

## 4. Coding & Architectural Standards
- **Interconnect**: Strictly use **AXI-Stream (AXIS)** for high-throughput radar data paths.
- **Memory**: Use DMA for PS-to-PL data transfer of the Saab AB dataset "Radar Cubes."
- **Logic**: Use Fixed-Point arithmetic. Avoid floating-point at all costs.
- **Verilog**: Use SystemVerilog where possible for complex state machines (Launch Authority FSM).
- **HLS**: Vitis HLS code must use `#pragma HLS PIPELINE` and `ARRAY_PARTITION` to meet real-time radar constraints.

## 5. Collaboration Protocol (The "Vanguard" Way)
- **Planning First**: Before generating code, provide a "Vibe-Check Implementation Plan."
- **Educational Context**: Explain the "Why" behind DSP choices (e.g., Why CORDIC over Multipliers for this phase shift?).
- **Git Awareness**: Ensure every major logic change is accompanied by a suggested Git commit message.
- **Novelty Hunter**: Propose "Elite" features like micro-Doppler classification or physical layer security tweaks to the beamformer.