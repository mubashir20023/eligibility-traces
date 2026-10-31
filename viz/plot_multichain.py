"""Fig-3-style pictures of the multi-chain task with every state coloured (as in Fig. 2 of the ET paper).

Rows = learners, columns = snapshots over training. Three figures per call:
  <out>_values.png   value estimate of every state      (0 .. 1, true values are ~0.95-0.99)
  <out>_errors.png   estimate minus true value          (blue = too low, red = too high)
  <out>_credit.png   which states ONE update at the bottleneck changes (normalised to the largest change);
                     an ordinary trace should light up only chain 0 (the path taken), an expected trace all chains.

Inputs: tabular npz from tabular_snapshots.py and/or deep value_snapshots.npz files (from --snapshot_every).
Usage:
  python plot_multichain.py --tabular tabular_m32_s1.npz \
      --deep "QRC λ=0=logs/<run dir>/value_snapshots.npz" "QRC-ET (Alg. 3)=logs/<run dir>/value_snapshots.npz" \
      --steps 2000 10000 30000 100000 --out m32_s1
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def positions(m, n):
    """state id -> (x, y): start left, m chains of n states, bottleneck, terminal square on the right."""
    S = 1 + m * n + 1
    pos = np.zeros((S, 2))
    mid = (m - 1) / 2
    pos[0] = (0, mid)
    for c in range(m):
        for i in range(n):
            pos[1 + c * n + i] = (1 + i, m - 1 - c)          # chain 0 at the top
    pos[S - 1] = (n + 1, mid)
    return pos


def draw(ax, vals, m, n, cmap, vmin, vmax, title=None):
    pos = positions(m, n)
    S = len(pos)
    for c in range(m):                                        # edges
        ys = m - 1 - c
        ax.plot([0, 1], [pos[0, 1], ys], color="0.85", lw=0.4, zorder=0)
        ax.plot([1, n], [ys, ys], color="0.85", lw=0.4, zorder=0)
        ax.plot([n, n + 1], [ys, pos[S - 1, 1]], color="0.85", lw=0.4, zorder=0)
    ax.plot([n + 1, n + 2], [pos[S - 1, 1]] * 2, color="0.6", lw=0.8, zorder=0)
    size = max(8, 260 / max(m, 1) ** 0.9)
    ax.scatter(pos[:, 0], pos[:, 1], c=vals, cmap=cmap, vmin=vmin, vmax=vmax, s=size * 6, edgecolors="0.3", linewidths=0.3, zorder=2)
    ax.add_patch(Rectangle((n + 1.75, pos[S - 1, 1] - 0.35 * max(1, m / 8)), 0.5, 0.7 * max(1, m / 8), color="black", zorder=2))
    ax.set_xlim(-0.6, n + 2.6); ax.set_ylim(-1, m)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("auto")
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8)


def load(args):
    rows = []                                                  # (name, steps, values, credit, true, m, n)
    if args.tabular:
        d = np.load(args.tabular, allow_pickle=True)
        for k, name in [("TD", "Tabular TD (tuned λ)"), ("ET-now", "Tabular ET, z applied now (Alg. 1)"),
                        ("ET-lag", "Tabular ET, z one step late (Alg. 3 lag)")]:
            if f"{k}_step" in d:
                rows.append((name, d[f"{k}_step"], d[f"{k}_values"], d[f"{k}_credit"], d["true"], int(d["chains"]), int(d["chain_len"])))
    for spec in args.deep or []:
        name, path = spec.rsplit("=", 1)
        d = np.load(path)
        rows.append((name, d["step"], d["values"], d["credit_agent"], d["true"], int(d["chains"]), int(d["chain_len"])))
    return rows


def pick(steps, want):
    i = int(np.searchsorted(steps, want, side="right")) - 1
    return max(0, min(i, len(steps) - 1))


def figure(rows, want, kind, out, title):
    nr, nc = len(rows), len(want)
    m = rows[0][5]
    fig, axes = plt.subplots(nr, nc, figsize=(2.3 * nc + 1.2, (0.9 + 0.07 * m) * nr + 0.6), squeeze=False)
    cmap, vmin, vmax, label = {"values": ("magma", 0, 1, "value estimate"),
                               "errors": ("RdBu_r", -0.3, 0.3, "estimate − true value"),
                               "credit": ("magma", 0, 1, "change from one update at the bottleneck (normalised)")}[kind]
    for r, (name, steps, values, credit, true, m_, n_) in enumerate(rows):
        for c, w in enumerate(want):
            i = pick(steps, w)
            if kind == "values":
                v = values[i]
            elif kind == "errors":
                v = values[i] - true
            else:
                v = np.abs(credit[i]) / (np.max(np.abs(credit[i])) + 1e-12)
            draw(axes[r, c], v, m_, n_, cmap, vmin, vmax, title=f"step {int(steps[i]):,}" if r == 0 else None)
        axes[r, 0].text(-0.08, 0.5, name, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    fig.colorbar(sm, ax=axes, shrink=0.6, label=label, pad=0.01)
    fig.suptitle(title, fontsize=10)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabular"); ap.add_argument("--deep", nargs="*")
    ap.add_argument("--steps", nargs="+", type=int, default=[2000, 10000, 30000, 100000])
    ap.add_argument("--out", default="multichain"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    rows = load(a)
    for kind, t in [("values", "Value estimates"), ("errors", "Value error (estimate − true)"),
                    ("credit", "Credit map: states changed by one update at the bottleneck")]:
        figure(rows, a.steps, kind, f"{a.out}_{kind}.png", f"{t} | {a.label}")
