#!/usr/bin/env python3
"""
Project Vanguard -- VanguardNet 1D CNN (Wider Architecture)
=============================================================
Input: [B, 16, 256] — 8 doppler_max + 8 doppler_idx channels, 256 range bins.
Architecture: Conv1d(16→32→64→64) + GAP + Dropout + FC(64→3).
~25K params, <30 KB INT8, fits Zynq-7020 BRAM.

WHY wider? The 8→16→32 model couldn't learn subtle THREAT/CLEAN
differences. 16→32→64→64 gives enough capacity for the pattern:
  - doppler_idx channels are CONSISTENT for THREAT (one Doppler bin)
  - doppler_idx channels are RANDOM for CLEAN (noise)
"""

import torch
import torch.nn as nn
import torch.quantization as quant


class VanguardNet(nn.Module):
    """
    Wider 1D CNN for 3-class radar threat classification.

    Input:  [B, 16, 256]  — 8 doppler_max + 8 doppler_idx, 256 range bins
    Output: [B, 3]        — logits for THREAT / JAMMER / CLEAN

    Architecture:
      Conv1d(16→32, k=7, s=2, p=3)  → BN → ReLU   ⇒ [B, 32, 128]
      Conv1d(32→64, k=5, s=2, p=2)  → BN → ReLU   ⇒ [B, 64, 64]
      Conv1d(64→64, k=3, s=2, p=1)  → BN → ReLU   ⇒ [B, 64, 32]
      Conv1d(64→64, k=3, s=1, p=1)  → BN → ReLU   ⇒ [B, 64, 32]
      GAP                                           ⇒ [B, 64]
      Dropout(0.3)
      FC(64→3)                                      ⇒ [B, 3]
    """

    def __init__(self, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 16 → 32, k=7, stride=2
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            # Block 2: 32 → 64, k=5, stride=2
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Block 3: 64 → 64, k=3, stride=2
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # Block 4: 64 → 64, k=3, stride=1 (increases receptive field)
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, n_classes)

        # Quantization stubs
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()

    def forward(self, x):
        """x: [B, 16, 256]"""
        x = self.quant(x)
        x = self.features(x)              # [B, 64, 32]
        x = self.gap(x).squeeze(-1)       # [B, 64]
        x = self.dropout(x)
        x = self.fc(x)                    # [B, 3]
        x = self.dequant(x)
        return x


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_summary(model):
    print("=" * 60)
    print("  VanguardNet — Wider 1D CNN Radar Threat Classifier")
    print("=" * 60)
    total, trainable = count_parameters(model)
    print(f"\n  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  INT8 weight size:     {total / 1024:.1f} KB")
    print(f"  BRAM utilization:     {total / (4.9 * 1024 * 1024 / 8) * 100:.2f}%")
    print(f"\n  {'Layer':<30} {'Shape':<20} {'Params':>10}")
    print(f"  {'-'*60}")
    for name, param in model.named_parameters():
        print(f"  {name:<30} {str(list(param.shape)):<20} {param.numel():>10,}")
    # MAC estimate
    # Block1: 16*32*128*7 = 458,752
    # Block2: 32*64*64*5  = 655,360
    # Block3: 64*64*32*3  = 393,216
    # Block4: 64*64*32*3  = 393,216
    # FC: 64*3 = 192
    total_macs = 458752 + 655360 + 393216 + 393216 + 192
    print(f"\n  Estimated MACs:       {total_macs:,}")
    print(f"  @ 125 MHz, 220 DSPs:  ~{total_macs / 220 / 125e6 * 1e6:.0f} us")
    print(f"  Target latency:       <1000 us")
    margin = 1000 / (total_macs / 220 / 125e6 * 1e6)
    print(f"  Margin:               {margin:.0f}x headroom")
    print("=" * 60)


if __name__ == "__main__":
    model = VanguardNet()
    model_summary(model)
    dummy = torch.randn(1, 16, 256)
    out = model(dummy)
    print(f"\n  Forward: {list(dummy.shape)} -> {list(out.shape)}")
    print(f"  Scores: {out.detach().numpy().flatten()}")
    print(f"  Predicted: {['THREAT', 'JAMMER', 'CLEAN'][out.argmax().item()]}")
