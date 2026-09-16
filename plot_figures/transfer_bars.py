#!/usr/bin/env python3


import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams

# ── font: Arial (Liberation Sans, metric-identical substitute) ──
for _f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf"):
    fm.fontManager.addfont(_f)
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── data (hardcoded) ─────────────────────────────────────────────────────────
groups  = ["v2.1", "v1.5", "XL", "LoRA", "TI"]
brisque = [80.3,  98.8,  45.4, 50.7, 70.8]
fdfr    = [0.97,  0.99,  0.69, 0.67, 0.91]
ism     = [0.07,  0.03,  0.56, 0.26, 0.03]
# ────────────────────────────────────────────────────────────────────────────

# normalised bar heights (taller = better for all three)
brisque_h = [v / 100 for v in brisque]
fdfr_h    = list(fdfr)
ism_h     = [1 - v for v in ism]       # lower ISM → taller bar

# ── original pastel fills, each with a darker same-hue frame + hatch ──
C_B, C_F, C_I = "#FACDAE", "#DEEAF6", "#CAB3DE"   # peach / pale blue / lavender
E_B, E_F, E_I = "#C67E3C", "#5E88C2", "#8368B0"   # darker same-hue: frame + hatch
H_B, H_F, H_I = "xxx", "///", "..."               # grid / diagonal / dots
rcParams["hatch.linewidth"] = 0.5

bar_w = 0.22
x = np.arange(len(groups))

fig, ax = plt.subplots(figsize=(7, 3.4))

# darker frame (linewidth) + thinner hatch lines, both in the same-hue edge colour
ax.bar(x - bar_w, brisque_h, bar_w, color=C_B, hatch=H_B, edgecolor=E_B,
       linewidth=1.0, label="BRISQUE", zorder=2)
ax.bar(x,         fdfr_h,    bar_w, color=C_F, hatch=H_F, edgecolor=E_F,
       linewidth=1.0, label="FDFR",    zorder=2)
ax.bar(x + bar_w, ism_h,     bar_w, color=C_I, hatch=H_I, edgecolor=E_I,
       linewidth=1.0, label="1-ISM",   zorder=2)

# ── value annotations ──
DX = 0.045   # horizontal outward offset for the left/right labels
for i in range(len(groups)):
    # nudge the outer two labels outward so the trio doesn't crowd (esp. v1.5)
    ax.text(x[i] - bar_w - DX, brisque_h[i] + 0.02, f"{brisque[i]:.1f}",
            ha="center", va="bottom", fontsize=11)
    ax.text(x[i],              fdfr_h[i]    + 0.02, f"{fdfr[i]:.2f}",
            ha="center", va="bottom", fontsize=11)
    ax.text(x[i] + bar_w + DX, ism_h[i]     + 0.02, f"{ism_h[i]:.2f}",
            ha="center", va="bottom", fontsize=11)

# ── dashed separator between cross-model / cross-method ──
sep = (x[2] + x[3]) / 2
ax.axvline(sep, color="#cccccc", linestyle="--", linewidth=1, zorder=1)

# ── section headers ──
ax.text(np.mean(x[:3]), 1.12, "Cross-model",  ha="center", fontsize=13.5)
ax.text(np.mean(x[3:]), 1.12, "Cross-method", ha="center", fontsize=13.5)

# ── axes ──
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=15)
ax.set_ylim(0, 1.32)

# remove y-axis tick labels and axis label; keep spine
ax.tick_params(axis="y", left=False, labelleft=False)
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── legend ──
ax.legend(loc="upper center", ncol=3, frameon=False, fontsize=12.5,
          bbox_to_anchor=(0.5, 1.08))

fig.tight_layout()

outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    out = os.path.join(outdir, f"transfer_bars.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved: {out}")
