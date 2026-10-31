"""Summarize R33 (one-hot input, sparse reward). Value-error AUC over the run, lower is better.
Per task: agent x learning-rate table (mean ± s.e. over 10 seeds), then each QRC-ET agent at its best learning rate
against the best QRC (any lambda, any learning rate), paired over seeds.
Usage (from stream-rep-rl):  python summarize_r33.py
"""
import csv, glob, re
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

TOTAL = 120_000
AGENTS = ["qrc_lam0", "qrc_lam09", "et_lag_default", "et_lag_b01", "et_nolag_b01", "et_nolag_b03"]
LRS = ["0.01", "0.03", "0.1"]
TASKS = ["m8-n4-p0.2", "m8-n4-p0.9", "m32-n4-p0.2", "m32-n4-p0.9"]
auc = defaultdict(dict)
for p in glob.glob("logs/multichain_env:MultiChainSparse-*__R33_*__*/training_data.csv"):
    m = re.search(r"MultiChainSparse-(m\d+-n\d+-p[\d.]+)-v0__R33_(.+)_lr([\d.]+)__(\d+)/", p)
    if not m:
        continue
    rows = list(csv.DictReader(open(p)))
    if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
        continue
    v = np.array([float(r["value_rmse"]) for r in rows])
    auc[(m.group(1), m.group(2), m.group(3))][int(m.group(4))] = float(np.mean(np.where(np.isfinite(v), v, 10.0)))

summary = []
for task in TASKS:
    print(f"\n=== {task}   (value-error AUC, mean ± s.e. [n]; lower = better)")
    print(f"{'agent':<16}" + "".join(f"{'lr ' + lr:>22}" for lr in LRS))
    best = {}
    for ag in AGENTS:
        cells = []
        for lr in LRS:
            d = auc.get((task, ag, lr), {})
            if d:
                v = np.array(list(d.values()))
                cells.append(f"{v.mean():.4f}±{v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.4f} [{len(v)}]")
                if ag not in best or v.mean() < best[ag][0]:
                    best[ag] = (v.mean(), lr)
            else:
                cells.append("-")
        print(f"{ag:<16}" + "".join(f"{c:>22}" for c in cells))
    q = [(best[a][0], a) for a in ("qrc_lam0", "qrc_lam09") if a in best]
    if not q:
        continue
    _, qa = min(q)
    qd = auc[(task, qa, best[qa][1])]
    for ag in AGENTS[2:]:
        if ag not in best:
            continue
        ed = auc[(task, ag, best[ag][1])]
        common = sorted(set(ed) & set(qd))
        if len(common) < 2:
            continue
        d = np.array([ed[s] - qd[s] for s in common])
        rel = 100 * d.mean() / np.mean([qd[s] for s in common])
        half = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.26) * d.std(ddof=1) / np.sqrt(len(d))
        pv = stats.ttest_rel([ed[s] for s in common], [qd[s] for s in common]).pvalue if stats else float("nan")
        line = (f"  {ag}@{best[ag][1]} vs best QRC ({qa}@{best[qa][1]}): {rel:+.1f}%  "
                f"diff {d.mean():+.4f} [{d.mean() - half:+.4f}, {d.mean() + half:+.4f}]  p={pv:.4f}  ET better on {int((d < 0).sum())}/{len(d)}")
        print(line)
        summary.append(f"{task:<12} {line.strip()}")
print("\nSUMMARY (negative % = QRC-ET has lower value error than the best-tuned QRC)")
for s in summary:
    print("  " + s)
print("Learning rates are chosen on the same seeds (screen). Confirm the winner on fresh seeds before concluding.")
