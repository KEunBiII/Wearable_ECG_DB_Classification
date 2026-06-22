"""
models.py - ECG Classification Model (A0: ResUNet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Block
# =============================================================================

class ResidualUBlock(nn.Module):
    """U-Net style Residual Block for 1D signals"""

    def __init__(self, out_ch: int, mid_ch: int, layers: int, downsampling: bool = True):
        super().__init__()
        self.downsample = downsampling
        K, P = 9, 4

        self.conv1 = nn.Conv1d(out_ch, out_ch, K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for idx in range(layers):
            in_ch = out_ch if idx == 0 else mid_ch
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, K, stride=2, padding=P, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.LeakyReLU()
            ))
            out_ch_dec = out_ch if idx == layers - 1 else mid_ch
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(mid_ch * 2, out_ch_dec, K, stride=2, padding=P,
                                   output_padding=1, bias=False),
                nn.BatchNorm1d(out_ch_dec),
                nn.LeakyReLU()
            ))

        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )

        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(2, 2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, 1, bias=False)

    def forward(self, x):
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        encoder_out = []
        out = x_in
        for layer in self.encoders:
            out = layer(out)
            encoder_out.append(out)

        out = self.bottleneck(out)

        for idx, layer in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(out, size=skip.size(-1), mode='linear', align_corners=False)
            out = layer(torch.cat([out, skip], dim=1))

        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]

        out += x_in

        if self.downsample:
            out = self.idfunc_1(self.idfunc_0(out))

        return out


# =============================================================================
# Model
# =============================================================================

class ResUNet(nn.Module):
    """A0: ECG only baseline with dense block + self-attention"""

    def __init__(self, nOUT, in_channels=1, out_ch=180, mid_ch=30):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_ch, 15, padding=7, stride=2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=4)
        self.rub_1 = ResidualUBlock(out_ch, mid_ch, layers=3)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch * 2),
            nn.LayerNorm(out_ch * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(out_ch * 2, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
            nn.Linear(out_ch, nOUT)
        )

    def forward(self, ecg_signal):
        x1 = F.leaky_relu(self.bn(self.conv(ecg_signal)))
        x2 = self.rub_0(x1)
        x3 = self.rub_1(x2)
        z = x3.mean(dim=-1)
        logits = self.fc(z)
        return logits, x1, x2, x3, z
