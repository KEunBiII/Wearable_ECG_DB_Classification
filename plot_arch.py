#!/usr/bin/env python3
"""
ResUNet Architecture Diagram
Usage:  python plot_arch.py
Output: arch_diagram.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def rbox(ax, cx, cy, w, h, text, fc, ec, fs=8.5, lw=1.5):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.06', facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3
    ))
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fs, fontweight='bold', zorder=4, linespacing=1.5)


def arr(ax, x1, y1, x2, y2, color='#333333', lw=1.5, ms=12):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=ms),
                zorder=5)


# ─── (a) Overall Architecture ─────────────────────────────────────────────────

def draw_overview(ax):
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.5, 23)
    ax.axis('off')
    ax.set_title('(a)  ResUNet — Overall Architecture', fontsize=12, fontweight='bold', pad=10)

    CX, W = 5.0, 8.5
    blocks = [
        (21.5, 0.7,  'Input ECG Beat  (B, 1, 501)',
         '#F0F0F0', '#888888'),
        (19.6, 1.6,  'Conv1d (k=15, stride=2)\nBatchNorm  +  LeakyReLU\n→ (B, 180, 251)',
         '#CCE5FF', '#1565C0'),
        (17.2, 1.8,  'ResidualUBlock  #1\nlayers=4,  downsampling=True\n→ (B, 180, 125)',
         '#FFE0B2', '#BF360C'),
        (14.8, 1.8,  'ResidualUBlock  #2\nlayers=3,  downsampling=True\n→ (B, 180, 62)',
         '#FFE0B2', '#BF360C'),
        (12.7, 1.0,  'Global Average Pooling  (dim=−1)\n→ (B, 180)',
         '#C8E6C9', '#1B5E20'),
        ( 9.8, 2.8,  'MLP Classifier\nLinear(180→360) + LayerNorm + GELU\nDropout(0.2)\nLinear(360→180) + LayerNorm + GELU\nLinear(180→nOUT)',
         '#F8BBD0', '#880E4F'),
        ( 7.7, 0.7,  'Output Logits  (B, nOUT)',
         '#F0F0F0', '#888888'),
    ]

    prev_bot = None
    for cy, h, text, fc, ec in blocks:
        rbox(ax, CX, cy, W, h, text, fc, ec, fs=8.5)
        if prev_bot is not None:
            arr(ax, CX, prev_bot, CX, cy + h/2)
        prev_bot = cy - h/2

    for cy, note in [(21.5, '501 samples @ 250 Hz ≈ 2 sec'),
                     (19.6, '÷2 temporal resolution'),
                     (17.2, '÷2 temporal resolution'),
                     (14.8, '÷2 temporal resolution')]:
        ax.text(9.7, cy, note, ha='left', va='center',
                fontsize=7, color='#555555', style='italic')


# ─── (b) ResidualUBlock Internal ──────────────────────────────────────────────

def draw_rub(ax):
    ax.set_xlim(0, 13)
    ax.set_ylim(-0.8, 23)
    ax.axis('off')
    ax.set_title('(b)  ResidualUBlock Internal  (layers = 4)',
                 fontsize=12, fontweight='bold', pad=10)

    CX, W = 5.5, 8.0
    HL, HR = CX - W/2, CX + W/2   # 1.5, 9.5
    BH = 0.95

    C_IO  = ('#F0F0F0', '#888888')
    C_ENC = ('#CCE5FF', '#1565C0')
    C_BOT = ('#C8E6C9', '#1B5E20')
    C_DEC = ('#FFE0B2', '#BF360C')
    C_ADD = ('#F8BBD0', '#880E4F')

    y_in, y_conv = 22.0, 20.5
    y_enc = [18.8, 17.1, 15.4, 13.7]
    y_bot = 11.8
    y_dec = [10.1,  8.4,  6.7,  5.0]
    y_add, y_pool = 3.3, 1.7

    # ── Boxes ─────────────────────────────────────────────────────────────────
    rbox(ax, CX, y_in,   W, BH, 'Input  x   (B, ch, L)', *C_IO,  fs=9)
    rbox(ax, CX, y_conv, W, BH, 'Conv1d(k=9) + BN    →   x_in', *C_ENC, fs=9)

    for ye, txt in zip(y_enc, [
            'Encoder 1:  Conv1d(stride=2) + BN + LeakyReLU\n→ (B, mid_ch, L/2)',
            'Encoder 2:  Conv1d(stride=2) + BN + LeakyReLU\n→ (B, mid_ch, L/4)',
            'Encoder 3:  Conv1d(stride=2) + BN + LeakyReLU\n→ (B, mid_ch, L/8)',
            'Encoder 4:  Conv1d(stride=2) + BN + LeakyReLU\n→ (B, mid_ch, L/16)',
    ]):
        rbox(ax, CX, ye, W, BH, txt, *C_ENC, fs=8)

    rbox(ax, CX, y_bot, W, BH, 'Bottleneck:  Conv1d + BN + LeakyReLU', *C_BOT, fs=9)

    for yd, txt in zip(y_dec, [
            'Decoder 4:  cat[x, skip4] → ConvTranspose1d + BN + LeakyReLU',
            'Decoder 3:  cat[x, skip3] → ConvTranspose1d + BN + LeakyReLU',
            'Decoder 2:  cat[x, skip2] → ConvTranspose1d + BN + LeakyReLU',
            'Decoder 1:  cat[x, skip1] → ConvTranspose1d + BN + LeakyReLU',
    ]):
        rbox(ax, CX, yd, W, BH, txt, *C_DEC, fs=8)

    rbox(ax, CX, y_add,  W, BH, 'Add  ( decoded  +  x_in )',               *C_ADD, fs=9)
    rbox(ax, CX, y_pool, W, BH, 'AvgPool1d(2) + Conv1d(1×1)\n[if downsampling=True]', *C_IO, fs=8.5)

    # ── Flow arrows ───────────────────────────────────────────────────────────
    chain = [y_in, y_conv] + y_enc + [y_bot] + y_dec + [y_add, y_pool]
    for ya, yb in zip(chain[:-1], chain[1:]):
        arr(ax, CX, ya - BH/2, CX, yb + BH/2)
    arr(ax, CX, y_pool - BH/2, CX, 0.1)
    ax.text(CX, -0.05, 'Output  (B, ch, L/2)',
            ha='center', va='top', fontsize=9, fontweight='bold')

    # ── Skip connections (right side, staggered) ──────────────────────────────
    # i=0: Enc1→Dec1 (longest skip, outermost line)
    # i=3: Enc4→Dec4 (shortest skip, innermost line)
    skip_colors = ['#CC8800', '#CC5500', '#CC2200', '#990000']
    for i in range(4):
        ye  = y_enc[i]
        yd  = y_dec[3 - i]           # Enc1→Dec1, Enc2→Dec2, Enc3→Dec3, Enc4→Dec4
        xr  = HR + 0.3 + (3 - i) * 0.5   # longest skip gets outermost line
        col = skip_colors[i]

        ax.plot([HR, xr], [ye, ye], color=col, lw=1.2, zorder=2)
        ax.plot([xr, xr], [ye, yd], color=col, lw=1.2, zorder=2)
        arr(ax, xr, yd, HR, yd, color=col, lw=1.2, ms=9)
        ax.text(xr + 0.08, (ye + yd) / 2, f'skip {i+1}',
                ha='left', va='center', fontsize=7, color=col,
                style='italic', rotation=90)

    # ── Residual connection: x_in → Add (left side) ──────────────────────────
    xL, rc = HL - 0.7, '#880E4F'
    ax.plot([HL, xL], [y_conv, y_conv], color=rc, lw=1.5, zorder=2)
    ax.plot([xL, xL], [y_conv, y_add],  color=rc, lw=1.5, zorder=2)
    arr(ax, xL, y_add, HL, y_add, color=rc, lw=1.5, ms=10)
    ax.text(xL - 0.1, (y_conv + y_add) / 2, 'residual\n(x_in)',
            ha='right', va='center', fontsize=7.5, color=rc, style='italic')


# ─── Main ─────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 15))
fig.patch.set_facecolor('white')
draw_overview(ax1)
draw_rub(ax2)
plt.tight_layout(pad=2.0)
plt.savefig('arch_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: arch_diagram.png")
