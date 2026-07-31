import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_pr_curve(curve, point, out="results/pr_curve.png"):
    d = os.path.dirname(out)
    if d: os.makedirs(d, exist_ok=True)
    rec = [c["recall"] for c in curve]
    prec = [c["precision"] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, "-o", ms=3, label="WBF ensemble (score sweep)")
    ax.scatter([point["recall"]], [point["precision"]], c="red", zorder=5,
               label="Rule-based integrated")
    ax.annotate(f'({point["recall"]:.3f}, {point["precision"]:.3f})',
                (point["recall"], point["precision"]),
                textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall: WBF ensemble vs. rule-based integration")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
