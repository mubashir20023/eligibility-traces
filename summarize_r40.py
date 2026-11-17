"""R40 summary: expected-trace advantage vs feature sharing on the sparse multi-chain (linear agent, fixed codes).

For every reward (p) and code (D, k):
  overlap          mean cosine between the codes of different chain states
  trace diversity  spread of the lambda-traces arriving at the shared state over the m chains, relative to their mean
                   (what an expected trace averages; same formula as the R32 diagnostics)
  Theta error      relative error of the BEST POSSIBLE linear Theta (visit-weighted least squares) in predicting the
                   expected past trace E[e_t - x(S_t) | S_t = s] from the code x(s): 0 for one-hot, > 0 once states share
  best QRC / best QRC-ET value error (mean value RMSE over the run, lower is better) and QRC-ET vs QRC paired over seeds.
Best settings at the edge of the tried grid are marked "(edge)".

Usage (from stream-rep-rl):  python summarize_r40.py                    (R40)
                             python summarize_r40.py R40 R40b           (pooled with the R40b fairness check)
                             python summarize_r40.py R40 R40b --figure  (also writes r40_sharing.png)
                             python summarize_r40.py R40 R40b R40c --figure  (adds theta on a one-hot state id)
"""
import csv, glob, re, sys
from collections import defaultdict
import numpy as np
import gymnasium as gym
import multichain_env  # noqa: F401  registers the environments
try:
    from scipy import stats
except ImportError:
    stats = None

GL = 0.99 * 0.9
FIGURE = "--figure" in sys.argv
PREFIXES = [a for a in sys.argv[1:] if not a.startswith("--")] or ["R40"]
PRE = "(?:" + "|".join(PREFIXES) + ")"


def code_stats(env_id):
    e = gym.make(env_id).unwrapped
    X = e.all_observations().reshape(e.n_states, -1)
    S, n, m = e.n_states, e.n, e.m
    traces, E, w = [], np.zeros_like(X), np.zeros(S)
    for c in range(m):
        z = np.zeros(X.shape[1])
        for s in [0] + [1 + c * n + i for i in range(n)] + [S - 1]:
            z = GL * z + X[s]
            E[s] += z - X[s]; w[s] += 1
        traces.append(z)
    traces = np.array(traces); mean = traces.mean(0)
    div = float(np.mean(np.sum((traces - mean) ** 2, 1)) / np.sum(mean ** 2))
    Xc = X[1:S - 1] / np.linalg.norm(X[1:S - 1], axis=1, keepdims=True)
    cos = Xc @ Xc.T
    overlap = float(cos[~np.eye(len(cos), dtype=bool)].mean())
    E /= w[:, None]; vis = w / w.sum(); sw = np.sqrt(vis)[:, None]
    theta, *_ = np.linalg.lstsq(sw * X, sw * E, rcond=None)
    R = X @ theta - E
    terr = float(np.sqrt((vis[:, None] * R ** 2).sum() / (vis[:, None] * E ** 2).sum()))
    return overlap, div, terr


auc = defaultdict(dict)
for p in glob.glob("logs/multichain_env:MultiChainCode*/training_data.csv"):
    m = re.search(rf"MultiChainCode(?:Id)?-(m32-n4-p[\d.]+-D(\d+)-k(\d+))-v0__{PRE}_(.+)_lr([\d.]+)__(\d+)/", p)
    rows = list(csv.DictReader(open(p))) if m else []
    if not rows or float(rows[-1]["global_step"]) < 0.99 * 120000:
        continue
    v = np.array([float(r["value_rmse"]) for r in rows])
    auc[(m.group(1), m.group(4), m.group(5))][int(m.group(6))] = float(np.mean(np.where(np.isfinite(v), v, 10.0)))

summary, fig_rows = [], []
order = lambda t: (t.split("-D")[0], -int(re.search(r"D(\d+)", t).group(1)))
for task in sorted({k[0] for k in auc}, key=order):
    overlap, div, terr = code_stats(f"MultiChainCode-{task}-v0")
    best, lrs = {}, defaultdict(set)
    for (t, ag, lr), d in auc.items():
        if t != task or len(d) < 2:
            continue
        fam = "oracle" if ag.startswith("et_oracle") else ("et" if ag.startswith("et") else "qrc")
        lrs[fam].add(float(lr))
        mu = np.mean(list(d.values()))
        if fam not in best or mu < best[fam][0]:
            best[fam] = (mu, ag, float(lr), d)
    print(f"\n=== {task}   overlap {overlap:.3f}   trace diversity {div:.3f}   best linear Theta error {terr:.1%}")
    for (t, ag, lr), d in sorted(auc.items(), key=lambda x: (x[0][1], float(x[0][2]))):
        if t == task:
            v = np.array(list(d.values()))
            print(f"  {ag:14s} lr {lr:6s} {v.mean():.4f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.4f} [{len(v)}]")
    if "qrc" not in best or "et" not in best:
        continue
    edge = lambda fam: " (edge)" if best[fam][2] in (min(lrs[fam]), max(lrs[fam])) else ""
    row = dict(task=task, p=task.split("-p")[1].split("-")[0], D=int(re.search(r"D(\d+)", task).group(1)),
               overlap=overlap, div=div, terr=terr)
    for fam, name in (("et", "ET (theta on features)"), ("oracle", "ET (theta on state id)")):
        if fam not in best:
            continue
        q, e = best["qrc"][3], best[fam][3]
        common = sorted(set(q) & set(e))
        if len(common) < 2:
            continue
        d = np.array([e[s] - q[s] for s in common])
        qm = np.mean([q[s] for s in common])
        rel = 100 * d.mean() / qm
        h = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.26) * d.std(ddof=1) / np.sqrt(len(d))
        sig = "*" if (d.mean() + h < 0 or d.mean() - h > 0) else " "
        line = (f"{task:20s} theta err {terr:5.1%}  best QRC {best['qrc'][1]}@{best['qrc'][2]:g}{edge('qrc')}  "
                f"{name}: {best[fam][1]}@{best[fam][2]:g}{edge(fam)}  vs QRC {rel:+6.1f}%{sig} better on {int((d < 0).sum())}/{len(d)}")
        print("  -> " + line)
        summary.append(line)
        row[fam] = (rel, 100 * (d.mean() - h) / qm, 100 * (d.mean() + h) / qm)
    fig_rows.append(row)
print(f"\nSUMMARY {'+'.join(PREFIXES)} (negative % = QRC-ET has lower value error; * = significant at 95%; "
      "(edge) = best learning rate at the edge of the tried grid)")
for s in summary:
    print("  " + s)

if FIGURE and fig_rows:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    codes = sorted({r["D"] for r in fig_rows}, reverse=True)
    label = {D: ("one-hot" if D == 130 else f"D={D}") for D in codes}
    x = np.arange(len(codes))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
    ax = axes[0]
    for p, col in (("0.2", "#1f77b4"), ("0.9", "#d62728")):
        rr = {r["D"]: r for r in fig_rows if r["p"] == p}
        for fam, ls, mk, name in (("et", "-", "o", "theta on features"), ("oracle", "--", "s", "theta on state id")):
            if not any(fam in rr.get(D, {}) for D in codes):
                continue
            ys = [rr[D][fam][0] if fam in rr.get(D, {}) else np.nan for D in codes]
            err = [[rr[D][fam][0] - rr[D][fam][1] if fam in rr.get(D, {}) else 0 for D in codes],
                   [rr[D][fam][2] - rr[D][fam][0] if fam in rr.get(D, {}) else 0 for D in codes]]
            ax.errorbar(x, ys, yerr=err, marker=mk, ls=ls, capsize=3, color=col, label=f"p={p}, {name}")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x); ax.set_xticklabels([label[D] for D in codes])
    ax.set_xlabel("feature code (more sharing to the right)")
    ax.set_ylabel("QRC-ET vs best QRC: value error (%)\n(below 0 = QRC-ET better)")
    ax.set_title("Expected-trace advantage vs feature sharing", fontsize=11)
    ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    first = {r["D"]: r for r in fig_rows}
    ax.plot(x, [100 * first[D]["terr"] for D in codes], "o-", color="#d62728", label="best linear theta: error in expected trace (%)")
    ax.set_ylabel("best linear theta error (%)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels([label[D] for D in codes]); ax.set_xlabel("feature code (more sharing to the right)")
    ax.grid(alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(x, [first[D]["div"] for D in codes], "s--", color="#1f77b4", label="trace diversity")
    ax2.plot(x, [first[D]["overlap"] for D in codes], "^:", color="#7f7f7f", label="feature overlap")
    ax2.set_ylabel("trace diversity / overlap")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="center right")
    ax.set_title("Each code: best linear theta error, trace diversity and overlap", fontsize=10)
    fig.tight_layout(); fig.savefig("r40_sharing.png", dpi=150)
    print("\nWrote r40_sharing.png")
