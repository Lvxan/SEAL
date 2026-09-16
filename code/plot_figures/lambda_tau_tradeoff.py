#!/usr/bin/env python3

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
from matplotlib import rcParams

for _f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSans-*.ttf"):
    fm.fontManager.addfont(_f)
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

outdir = os.path.dirname(os.path.abspath(__file__))

# ── data (hardcoded; CelebA-HQ, prompt P1). Each entry: x -> (LPIPS, ISM) ────
LAMBDA = {   # dispersion weight lambda
    0.0:  (0.06, 0.02),
    0.1:  (0.07, 0.24),
    0.25: (0.07, 0.21),
    0.5:  (0.07, 0.06),
    1.0:  (0.07, 0.10),
}
TAU = {      # density threshold tau
    0.05: (0.07, 0.05),
    0.1:  (0.07, 0.08),
    0.2:  (0.07, 0.15),
    0.3:  (0.07, 0.14),
    0.4:  (0.07, 0.13),
    0.5:  (0.07, 0.13),
}
# ────────────────────────────────────────────────────────────────────────────

# palette identical to steps_tradeoff / ratio_tradeoff
C_LPIPS, C_ISM, C_ISM_FACE = "#E98F90", "#B49BD6", "#CAB3DE"


def make(table, xlabel, default, ymax, band, callout, callout_xy,
         callout_xytext, out_name):
    xs = list(table)
    lpips = [table[x][0] for x in xs]
    ism = [table[x][1] for x in xs]
    fig, ax1 = plt.subplots(figsize=(5.0, 2.49))
    ax1.axvspan(default - band, default + band, facecolor="#E4F2E0",
                edgecolor="#C7E3C2", linewidth=0.8, zorder=0)
    ax1.axvline(x=default, color="#9AA39A", linestyle=":", linewidth=1.1, zorder=1)
    ln1 = ax1.plot(xs, lpips, "o-", color=C_LPIPS, linewidth=2.0, markersize=9,
                   markerfacecolor=C_LPIPS, markeredgecolor="white",
                   markeredgewidth=1.1, label="LPIPS", zorder=4)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(r"LPIPS $\downarrow$")
    ax1.set_ylim(0, ymax); ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax2 = ax1.twinx()
    ln2 = ax2.plot(xs, ism, "s--", color=C_ISM, linewidth=2.0, markersize=9,
                   markerfacecolor=C_ISM_FACE, markeredgecolor="white",
                   markeredgewidth=1.1, label="ISM", zorder=4)
    ax2.set_ylabel(r"ISM $\downarrow$")
    ax2.set_ylim(0, ymax); ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax1.set_xticks(xs); ax1.set_xticklabels([f"{x:g}" for x in xs])
    ax1.annotate(callout, xy=callout_xy, xytext=callout_xytext, fontsize=11,
                 ha="center", va="center", color="#3A3A3A",
                 arrowprops=dict(arrowstyle="-", color="#8A8A8A", lw=1.0,
                                 connectionstyle="arc3,rad=0.3"),
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor="#9EC79A", linewidth=1.2), zorder=5)
    lns = ln1 + ln2
    ax1.legend(lns, [l.get_label() for l in lns], loc="upper right",
               frameon=True, framealpha=0.95, edgecolor="#CCCCCC", fontsize=10)
    ax1.spines["top"].set_visible(False); ax2.spines["top"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(outdir, f"{out_name}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight"); print("saved:", out)
    plt.close(fig)


# lambda: default 0.5 is a swept point
make(LAMBDA, r"Dispersion weight $\lambda$", 0.5, 0.3, 0.02,
     r"default $\lambda{=}0.5$", (0.5, 0.075), (0.78, 0.155), "lambda_tradeoff")

# tau: default 0.15 lies between swept points
make(TAU, r"Density threshold $\tau$", 0.15, 0.3, 0.008,
     r"default $\tau{=}0.15$", (0.15, 0.115), (0.16, 0.245), "tau_tradeoff")
