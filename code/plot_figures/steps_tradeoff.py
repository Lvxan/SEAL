#!/usr/bin/env python3

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# ── data (hardcoded; CelebA-HQ, prompt P1) ──────────────────────────────────
steps      = [50, 100, 200, 300, 400]              # optimization steps T
lpips_vals = [0.08, 0.07, 0.07, 0.07, 0.07]        # LPIPS (perceptual cost, ↓)
ism_vals   = [0.65, 0.66, 0.20, 0.07, 0.11]        # ISM   (protection,      ↓)
# ────────────────────────────────────────────────────────────────────────────

# ── colours (matched to ratio_tradeoff.pdf) ──
C_LPIPS    = "#E98F90"   # salmon
C_ISM      = "#B49BD6"   # lavender line
C_ISM_FACE = "#CAB3DE"   # lavender marker fill

YMAX = 0.7
DEFAULT = 300

fig, ax1 = plt.subplots(figsize=(5.0, 2.49))

# ── default T=300 highlight band ──
ax1.axvspan(285, 315, facecolor="#E4F2E0", edgecolor="#C7E3C2",
            linewidth=0.8, zorder=0)
ax1.axvline(x=DEFAULT, color="#9AA39A", linestyle=":", linewidth=1.1, zorder=1)

# ── left axis: LPIPS ──
ln1 = ax1.plot(steps, lpips_vals, "o-", color=C_LPIPS, linewidth=2.0,
               markersize=9, markerfacecolor=C_LPIPS,
               markeredgecolor="white", markeredgewidth=1.1,
               label="LPIPS", zorder=4)
ax1.set_ylabel(r"LPIPS $\downarrow$")
ax1.set_ylim(0, YMAX)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# ── right axis: ISM ──
ax2 = ax1.twinx()
ln2 = ax2.plot(steps, ism_vals, "s--", color=C_ISM, linewidth=2.0,
               markersize=9, markerfacecolor=C_ISM_FACE,
               markeredgecolor="white", markeredgewidth=1.1,
               label="ISM", zorder=4)
ax2.set_ylabel(r"ISM $\downarrow$")
ax2.set_ylim(0, YMAX)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

# ── x ticks ──
ax1.set_xticks(steps)
ax1.set_xticklabels([str(t) for t in steps])

# ── "default T=300" callout ──
ax1.annotate(r"default $T{=}300$",
             xy=(DEFAULT, 0.10), xytext=(230, 0.50),
             fontsize=11, ha="center", va="center", color="#3A3A3A",
             arrowprops=dict(arrowstyle="-", color="#8A8A8A", lw=1.0,
                             connectionstyle="arc3,rad=0.3"),
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#9EC79A", linewidth=1.2), zorder=5)

# ── combined legend ──
lns = ln1 + ln2
ax1.legend(lns, [l.get_label() for l in lns], loc="upper right",
           frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=10)

ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)

fig.tight_layout()
outdir = os.path.dirname(os.path.abspath(__file__))
for ext in ("pdf", "png"):
    out = os.path.join(outdir, f"steps_tradeoff.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"saved: {out}")
