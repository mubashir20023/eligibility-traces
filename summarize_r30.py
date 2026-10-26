"""Summarize R30 (multi-chain prediction screen). Lower is better everywhere.

Per run: AUC = mean value_rmse over the whole run (learning speed + accuracy),
FINAL = mean value_rmse over the last 10% of steps.
Per environment: an agent x learning-rate table of AUC, then best ET vs the best QRC over all lambdas.
Usage (from stream-rep-rl):  python summarize_r30.py
"""
import csv, glob, math, re
from collections import defaultdict
import numpy as np

TOTAL = 100_000
AGENTS = ["qrc_lam0", "qrc_lam024", "qrc_lam08", "et_eta0", "et_eta03"]
LRS = ["3e-4", "1e-3", "3e-3"]
ENVS = [f"m{m}-n4-s{s}" for s in ("0.0", "1.0") for m in (1, 8, 32, 128)]

runs = defaultdict(dict)   # (env, agent, lr) -> {seed: (auc, final, bad)}
for p in glob.glob("logs/multichain_env:MultiChainPred-*__R30_*__*/training_data.csv"):
    m = re.search(r"MultiChainPred-(m\d+-n\d+-s[\d.]+)-v0__R30_(.+)_lr([\de.-]+)__(\d+)/", p)
    if not m:
        continue
    env, agent, lr, seed = m.group(1), m.group(2), m.group(3), int(m.group(4))
    with open(p) as f:
        rows = list(csv.DictReader(f))
    if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
        continue
    steps = np.array([float(r["global_step"]) for r in rows])
    err = np.array([float(r["value_rmse"]) for r in rows])
    bad = not np.all(np.isfinite(err))
    err = np.where(np.isfinite(err), err, 10.0)          # a diverged run counts as error 10
    runs[(env, agent, lr)][seed] = (err.mean(), err[steps > 0.9 * TOTAL].mean(), bad)


def mean_sem(d, i):
    v = np.array([x[i] for x in d.values()])
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)


summary = []
for env in ENVS:
    print(f"\n=== {env}   (cells: value-RMSE AUC mean±sem [n, diverged]; lower = better)")
    print(f"{'agent':<12}" + "".join(f"{lr:>22}" for lr in LRS) + f"{'best AUC (final)':>24}")
    best = {}
    for agent in AGENTS:
        cells, opts = [], []
        for lr in LRS:
            d = runs.get((env, agent, lr), {})
            if not d:
                cells.append("-"); continue
            mu, se = mean_sem(d, 0)
            cells.append(f"{mu:.4f}±{se:.4f} [{len(d)},{sum(x[2] for x in d.values())}]")
            opts.append((mu, lr))
        if opts:
            b = min(opts); best[agent] = b
            cells.append(f"{b[0]:.4f} ({mean_sem(runs[(env, agent, b[1])], 1)[0]:.4f}) @{b[1]}")
        print(f"{agent:<12}" + "".join(f"{c:>22}" for c in cells))
    qrc = [(best[a][0], a, best[a][1]) for a in AGENTS[:3] if a in best]
    for et in ["et_eta0", "et_eta03"]:
        if et not in best or not qrc:
            continue
        _, qa, qlr = min(qrc)
        a, b = runs[(env, et, best[et][1])], runs[(env, qa, qlr)]
        common = sorted(set(a) & set(b))
        if len(common) < 2:
            continue
        d = np.array([a[s][0] - b[s][0] for s in common])
        rel = 100 * d.mean() / np.mean([b[s][0] for s in common])
        print(f"  {et}@{best[et][1]} minus best QRC ({qa}@{qlr}): {d.mean():+.4f} ({rel:+.1f}%), "
              f"sem {d.std(ddof=1) / np.sqrt(len(d)):.4f}, ET lower on {int((d < 0).sum())}/{len(d)} seeds")
        summary.append((env, et, rel, int((d < 0).sum()), len(d)))

print("\nSUMMARY: ET vs best-tuned QRC (negative % = ET has lower value error)")
for env, et, rel, wins, n in summary:
    print(f"  {env:<14} {et:<9} {rel:+6.1f}%   ET better on {wins}/{n} seeds")
print("Best settings are picked on the same 5 seeds (screen only). Confirm on fresh seeds before concluding.")
