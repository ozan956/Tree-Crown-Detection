"""Publication-quality traditional-pipeline workflow diagram (paper Fig. 1).

Rebuilds the v1 workflow with the faithful feature names (illumination-invariant
color + Gabor filter bank) and a clean, print-ready style. Vector PDF output.

Run: .venv/bin/python -m improvement.make_workflow
Writes media/fig_workflow.pdf and media/fig_workflow.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

INK = "#222222"
BOX = "#f4f6f8"
EDGE = "#0072B2"
ACCENT = "#D55E00"


def box(ax, x, y, w, h, text, fc=BOX, ec=EDGE, fs=8.5):
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle="round,pad=0.012,rounding_size=0.03",
                       linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, zorder=5)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=11, linewidth=1.1, color=color, shrinkA=1, shrinkB=1))


def main():
    fig, ax = plt.subplots(figsize=(13, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    # input
    box(ax, 7, 17, 11, 6, "Satellite\nImage", fc="#eef2f6")
    # two features (faithful names)
    box(ax, 24, 24, 15, 6.5, "Illumination-Invariant\nColor Feature")
    box(ax, 24, 10, 15, 6.5, "Gabor Filter Bank\n(4 orient. × 5 scale)")
    arrow(ax, 12.5, 17, 16.5, 22.5); arrow(ax, 12.5, 17, 16.5, 11.5)
    # joint map column
    box(ax, 43, 24, 15, 6.5, "Joint Probability\nMap  J = w₁C + w₂G")
    box(ax, 43, 15.5, 15, 5.5, "Otsu Thresholding")
    box(ax, 43, 7, 15, 6, "Opening\n(remove low veg.)")
    arrow(ax, 31.5, 24, 35.5, 24); arrow(ax, 31.5, 10, 38, 8)
    arrow(ax, 43, 20.75, 43, 18.25); arrow(ax, 43, 12.75, 43, 10)
    # watershed column
    box(ax, 62, 24, 15, 6.5, "Watershed\nSegmentation")
    box(ax, 62, 15.5, 15, 6, "Distance Transform\n+ Local Maxima")
    box(ax, 62, 7, 15, 5.5, "Merge Maxima\n(crown-scale)")
    arrow(ax, 50.5, 22, 54.5, 24); arrow(ax, 62, 20.75, 62, 18.5); arrow(ax, 62, 12.5, 62, 9.75)
    # output
    box(ax, 84, 15.5, 16, 7.5, "Tree Centers &\nApproximate\nBoundaries", fc="#fdeee6", ec=ACCENT, fs=9)
    arrow(ax, 69.5, 9, 76, 13.5); arrow(ax, 69.5, 15.5, 76, 15.5)

    # feature-bracket label
    ax.text(24, 30.5, "Feature extraction", ha="center", fontsize=8.5, style="italic", color=EDGE)
    ax.text(52.5, 30.5, "Joint probability map → segmentation", ha="center", fontsize=8.5, style="italic", color=EDGE)

    fig.tight_layout()
    fig.savefig("media/fig_workflow.pdf", bbox_inches="tight")
    fig.savefig("media/fig_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote media/fig_workflow.{pdf,png}")


if __name__ == "__main__":
    main()
