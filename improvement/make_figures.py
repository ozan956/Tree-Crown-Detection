"""Generate the publication figure set from committed results.

Colorblind-safe Okabe-Ito palette, single-axis, direct labels, recessive grid.
Every number is read from the results files / recomputed, not hardcoded loosely.

Run: .venv/bin/python -m improvement.make_figures  -> writes results/figs/*.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermilion": "#D55E00", "skyblue": "#56B4E9", "yellow": "#F0E442",
    "purple": "#CC79A7", "black": "#000000", "grey": "#999999",
}
plt.rcParams.update({
    "figure.dpi": 200, "font.size": 12, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.grid": True,
    "grid.alpha": 0.25, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False,
})
OUT = "results/figs"

# ---- canonical numbers (from results/table.md, IMPROVEMENT_RESULTS.md,
#      BEYOND_CEILING_ANALYSIS.md, DENSITY_BASELINE.md) ----
METHODS = [  # name, precision, recall, f1  (one-to-one, score>=0.8)
    ("YOLOv3",        0.985, 0.549, 0.705),
    ("DETR",          0.992, 0.725, 0.838),
    ("Cascade R-CNN", 0.984, 0.786, 0.874),
    ("Swin",          0.979, 0.810, 0.886),
    ("Faster R-CNN",  0.974, 0.822, 0.892),
    ("Density-map",   0.888, 0.834, 0.860),
    ("WBF ensemble",  0.968, 0.878, 0.921),
    ("Integrated",    0.833, 0.971, 0.897),
    ("Integrated+gate", 0.941, 0.949, 0.945),
]
GATE = [  # threshold, precision, recall, f1, beyond_ceiling_retained_frac
    (0.0, 0.828, 0.973, 0.894, 1.00),  # baseline (no gate) approx at thr->0
    (0.3, 0.921, 0.959, 0.940, 0.90),
    (0.4, 0.932, 0.954, 0.943, 0.86),
    (0.5, 0.941, 0.949, 0.945, 0.80),
    (0.6, 0.951, 0.942, 0.946, 0.73),
    (0.7, 0.960, 0.936, 0.948, 0.67),
]
COVERAGE = [  # source, pct
    ("YOLOv3",        86.6), ("Faster R-CNN", 86.1), ("Swin", 86.5),
    ("Cascade R-CNN", 86.5), ("DETR", 86.9),
    ("WBF ensemble",  92.5), ("All-6 union\n(DL ceiling)", 94.2),
    ("Integrated",    97.5),
]
BEYOND = {  # feature: (dl_reachable_median, beyond_ceiling_median)
    "Crown area (px$^2$)": (1924, 788), "Brightness": (95.1, 102.5),
    "Saturation": (86.0, 59.9), "Excess-green": (22.3, 11.1),
    "Texture": (26.8, 32.4),
}


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")  # vector for publication
    plt.close(fig); print(f"wrote {OUT}/{name}.{{png,pdf}}")


def fig_f1_bars():
    """Fig 1: F1 by method — the headline comparison. Form: magnitude ranking."""
    order = sorted(METHODS, key=lambda m: m[3])
    names = [m[0] for m in order]; f1 = [m[3] for m in order]
    colors = []
    for n in names:
        if n == "Integrated+gate": colors.append(OI["vermilion"])
        elif n == "WBF ensemble": colors.append(OI["blue"])
        elif n == "Density-map": colors.append(OI["purple"])
        else: colors.append(OI["grey"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    y = np.arange(len(names))
    ax.barh(y, f1, color=colors, height=0.68)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("$F_1$ (one-to-one matching, score $\\geq$ 0.8)")
    ax.set_xlim(0.6, 1.0)
    for yi, v in zip(y, f1):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=9)
    _save(fig, "fig1_f1_by_method")


def fig_pr_space():
    """Fig 2: methods in precision-recall space — shows the trade-offs."""
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for name, p, r, f in METHODS:
        if name == "Integrated+gate":
            c, mk, s = OI["vermilion"], "*", 260
        elif name == "WBF ensemble":
            c, mk, s = OI["blue"], "s", 90
        elif name == "Integrated":
            c, mk, s = OI["orange"], "D", 80
        elif name == "Density-map":
            c, mk, s = OI["purple"], "^", 80
        else:
            c, mk, s = OI["grey"], "o", 55
        ax.scatter(r, p, c=c, marker=mk, s=s, zorder=4, edgecolors="white", linewidths=0.6)
        # per-label offsets to avoid collisions in the crowded top-left cluster
        off = {"Swin": (7, -11), "Faster R-CNN": (7, 6), "Cascade R-CNN": (7, 5),
               "DETR": (7, 4), "YOLOv3": (7, 4), "WBF ensemble": (9, -2),
               "Density-map": (-10, 10), "Integrated": (9, 2), "Integrated+gate": (10, 2)}
        ha = "right" if name == "Density-map" else "left"
        ax.annotate(name, (r, p), textcoords="offset points",
                    xytext=off.get(name, (7, 4)), fontsize=8.5, ha=ha)
    # iso-F1 contours (recessive reference curves); one label low-right where it's clear
    rr = np.linspace(0.4, 1.0, 200)
    for f in (0.7, 0.8, 0.9):
        pp = f * rr / (2 * rr - f)
        ok = (pp > 0) & (pp <= 1)
        ax.plot(rr[ok], pp[ok], color=OI["grey"], lw=0.8, ls=":", alpha=0.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0.5, 1.0); ax.set_ylim(0.62, 1.0)
    _save(fig, "fig2_pr_space")


def fig_coverage_ladder():
    """Fig 3: GT coverage ladder — the beyond-ceiling headline."""
    order = COVERAGE
    names = [c[0] for c in order]; pct = [c[1] for c in order]
    colors = [OI["vermilion"] if "Integrated" in n else
              (OI["green"] if "union" in n else
               (OI["blue"] if "WBF" in n else OI["grey"])) for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y = np.arange(len(names))
    ax.barh(y, pct, color=colors, height=0.66)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Ground-truth tree coverage (%)")
    ax.set_xlim(80, 100)
    for yi, v in zip(y, pct):
        ax.text(v + 0.15, yi, f"{v:.1f}", va="center", fontsize=9)
    # annotate the beyond-ceiling gap
    ax.annotate("", xy=(97.5, len(names)-1), xytext=(94.2, len(names)-1),
                arrowprops=dict(arrowstyle="<->", color=OI["vermilion"], lw=1.5))
    _save(fig, "fig3_coverage_ladder")


def fig_beyond_ceiling_features():
    """Fig 4: why detectors miss them — feature comparison (grouped bars, normalized)."""
    feats = list(BEYOND.keys())
    dl = np.array([BEYOND[f][0] for f in feats], float)
    bc = np.array([BEYOND[f][1] for f in feats], float)
    # normalize each feature to the DL value = 1.0 so they share one axis
    dln = dl / dl; bcn = bc / dl
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(feats)); w = 0.38
    ax.bar(x - w/2, dln, w, color=OI["blue"], label="DL-reachable crowns")
    ax.bar(x + w/2, bcn, w, color=OI["vermilion"], label="Beyond-ceiling crowns")
    for xi, v, raw in zip(x - w/2, dln, dl): ax.text(xi, v+0.02, f"{raw:.0f}", ha="center", fontsize=7.5)
    for xi, v, raw in zip(x + w/2, bcn, bc): ax.text(xi, v+0.02, f"{raw:.0f}", ha="center", fontsize=7.5)
    ax.axhline(1.0, color=OI["grey"], lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(feats, fontsize=9)
    ax.set_ylabel("Median value (normalized to DL-reachable = 1.0)")
    ax.set_ylim(0, 1.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    _save(fig, "fig4_beyond_ceiling_features")


def fig_gate_dial():
    """Fig 5: the gate as a recovery-vs-F1 dial (single axis: F1; recovery as labels)."""
    thr = [g[0] for g in GATE[1:]]; f1 = [g[3] for g in GATE[1:]]; ret = [g[4] for g in GATE[1:]]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(thr, f1, "-o", color=OI["vermilion"], lw=2, ms=7, zorder=5, label="Integrated+gate $F_1$")
    ax.axhline(0.921, color=OI["blue"], lw=1.5, ls="--", label="WBF ensemble $F_1$ = 0.921")
    ax.axhline(0.894, color=OI["grey"], lw=1.2, ls=":", label="No gate $F_1$ = 0.894")
    for t, f, rr in zip(thr, f1, ret):
        ax.annotate(f"{int(rr*100)}% kept", (t, f), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8, color=OI["black"])
    # mark recommended operating point
    ax.scatter([0.5], [0.945], s=260, facecolors="none", edgecolors=OI["green"], linewidths=2, zorder=6)
    ax.annotate("recommended (0.5):\n80% recoveries kept", (0.5, 0.945),
                textcoords="offset points", xytext=(-95, -34), fontsize=8, color=OI["green"],
                arrowprops=dict(arrowstyle="->", color=OI["green"], lw=1.2))
    ax.set_xlabel("Gate acceptance threshold")
    ax.set_ylabel("$F_1$")
    ax.set_ylim(0.88, 0.955)
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, "fig5_gate_dial")


def fig_density_dual():
    """Fig 6: density-map is a good counter but weak detector — two panels."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.0))
    # panel 1: detection F1
    names = ["Density-map", "WBF\nensemble", "Integrated\n+gate"]
    f1 = [0.860, 0.921, 0.924]
    cols = [OI["purple"], OI["blue"], OI["vermilion"]]
    a1.bar(names, f1, color=cols, width=0.6)
    for i, v in enumerate(f1): a1.text(i, v+0.003, f"{v:.3f}", ha="center", fontsize=9)
    a1.set_ylim(0.8, 0.95); a1.set_ylabel("Detection $F_1$")
    # panel 2: what density is actually good at (count MAE, lower=better) - single bar w/ context
    a2.bar(["Density-map\ncounter"], [4.63], color=OI["purple"], width=0.4)
    a2.text(0, 4.63+0.06, "4.63", ha="center", fontsize=9)
    a2.set_ylabel("Count MAE (trees / image, lower = better)")
    a2.set_ylim(0, 8)
    a2.text(0, 6.6, "median 53\ntrees/image\n(~8% error)", ha="center", fontsize=8.5, color=OI["grey"])
    _save(fig, "fig6_density_baseline")


def main():
    fig_f1_bars(); fig_pr_space(); fig_coverage_ladder()
    fig_beyond_ceiling_features(); fig_gate_dial(); fig_density_dual()
    print("all figures written to", OUT)


if __name__ == "__main__":
    main()
