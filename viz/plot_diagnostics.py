"""Diagnostic curves: why expected traces cannot change anything in the deep agents on the multi-chain task.

Reads the diag_* arrays written by apply_snapshot_patch.py (value_snapshots_episodes.npz) and plots, against episode:
  trace diversity    how different the traces arriving at the shared state are across chains, relative to their mean
                     (the room an expected trace has to differ from a sampled one; tabular reference drawn dashed)
  feature cosine     mean cosine similarity of the last-layer features of different states (1 = identical)
  effective rank     effective rank of the feature matrix
  kernel spread      how much one update at the shared state moves the other states, relative to itself (1 = all equal)
  z accuracy         cosine and length ratio of the learned expected trace vs the true expected trace (ET runs only)
Usage: python plot_diagnostics.py --run "QRC λ=0.8=<path>.npz" --run "QRC-ET=<path>.npz" --out diagnostics.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})


def tabular_diversity(m, n=4, gl=0.99 * 0.8):
    """Trace diversity at the shared state for one-hot features (closed form)."""
    chain = sum(gl ** (2 * k) for k in range(1, n + 1))
    return chain * (1 - 1 / m) / (1 + chain / m + gl ** (2 * (n + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help='"name=path/value_snapshots_episodes.npz"')
    ap.add_argument("--lam", type=float, default=0.8)
    ap.add_argument("--out", default="diagnostics.png")
    a = ap.parse_args()
    runs = []
    for spec in a.run:
        name, path = spec.rsplit("=", 1)
        runs.append((name, np.load(path)))
    m = int(runs[0][1]["chains"])
    panels = [("diag_trace_diversity", "Trace diversity at shared state", None),
              ("diag_feature_cosine", "Feature cosine between states", (0, 1.05)),
              ("diag_feature_eff_rank", "Effective rank of features", None),
              ("diag_kernel_spread", "Kernel spread (1 = every state moves equally)", (0, 1.1)),
              ("diag_z_cosine", "ET: cosine(z, true expected trace)", (0, 1.05)),
              ("diag_z_norm_ratio", "ET: |z| / |true expected trace|", (0, 1.2))]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
    for ax, (key, title, ylim) in zip(axes.flat, panels):
        for name, d in runs:
            if key in d.files and np.isfinite(d[key]).any():
                ax.plot(d["episode"], d[key], marker="o", ms=3.5, lw=1.4, label=name)
        if key == "diag_trace_diversity":
            ax.axhline(tabular_diversity(m, gl=0.99 * a.lam), color="black", ls="--", lw=1, label="tabular (one-hot) features")
            ax.set_yscale("symlog", linthresh=0.01)
        if key == "diag_z_norm_ratio":
            ax.axhline(1.0, color="black", ls=":", lw=1)
        ax.set_xscale("log"); ax.set_xlabel("episode"); ax.set_title(title, fontsize=10); ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Why expected traces cannot help here | multi-chain, {m} chains, reward 1 + N(0,1)", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out, dpi=180, bbox_inches="tight")
    print("saved", a.out)


if __name__ == "__main__":
    main()
