"""Summarize any multi-chain prediction job (R33, R34a, R34b, ...). Value-error AUC over the run, lower is better.

Finds logs/multichain_env:<Env>-<task>-v0__<PREFIX>_<agent>_lr<lr>__<seed>/training_data.csv, then per task prints an
agent x learning-rate table and compares every QRC-ET agent (at its best learning rate) with the best QRC agent
(any lambda, any learning rate), paired over seeds.

Usage (from stream-rep-rl):  python summarize_mc.py R34a     |   python summarize_mc.py R34a R35a  (several jobs pooled)
"""
import csv, glob, re, sys
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

prefixes = sys.argv[1:] or ["R33"]
prefix = "+".join(prefixes)
PRE = "(?:" + "|".join(prefixes) + ")"
auc, steps_max = defaultdict(dict), defaultdict(list)
for p in glob.glob("logs/multichain_env:*/training_data.csv"):
    m = re.search(rf"multichain_env:(\w+)-(m\d+-n\d+-[ps][\d.]+)-v0__{PRE}_(.+)_lr([\d.e-]+)__(\d+)/", p)
    if not m:
        continue
    rows = list(csv.DictReader(open(p)))
    if not rows or "value_rmse" not in rows[0]:
        continue
    task, agent, lr, seed = f"{m.group(1)} {m.group(2)}", m.group(3), m.group(4), int(m.group(5))
    steps_max[task].append(float(rows[-1]["global_step"]))
    v = np.array([float(r["value_rmse"]) for r in rows])
    auc[(task, agent, lr)][seed] = (float(rows[-1]["global_step"]), float(np.mean(np.where(np.isfinite(v), v, 10.0))))

summary = []
for task in sorted({k[0] for k in auc}):
    full = max(steps_max[task])
    data = {k[1:]: {s: a for s, (st, a) in d.items() if st >= 0.99 * full} for k, d in auc.items() if k[0] == task}
    agents = sorted({a for a, _ in data}, key=lambda a: (not a.startswith("qrc"), a))
    lrs = sorted({lr for _, lr in data}, key=float)
    print(f"\n=== {task}   (value-error AUC, mean ± s.e. [n]; lower = better)")
    print(f"{'agent':<22}" + "".join(f"{'lr ' + lr:>22}" for lr in lrs))
    best = {}
    for ag in agents:
        cells = []
        for lr in lrs:
            d = data.get((ag, lr), {})
            if d:
                v = np.array(list(d.values()))
                cells.append(f"{v.mean():.4f}±{v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.4f} [{len(v)}]")
                if ag not in best or v.mean() < best[ag][0]:
                    best[ag] = (v.mean(), lr)
            else:
                cells.append("-")
        print(f"{ag:<22}" + "".join(f"{c:>22}" for c in cells))
    q = [(best[a][0], a) for a in agents if a.startswith("qrc") and a in best]
    if not q:
        continue
    _, qa = min(q)
    qd = data[(qa, best[qa][1])]
    for ag in agents:
        if ag.startswith("qrc") or ag not in best:
            continue
        ed = data[(ag, best[ag][1])]
        common = sorted(set(ed) & set(qd))
        if len(common) < 2:
            continue
        d = np.array([ed[s] - qd[s] for s in common])
        rel = 100 * d.mean() / np.mean([qd[s] for s in common])
        half = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.1) * d.std(ddof=1) / np.sqrt(len(d))
        pv = stats.ttest_rel([ed[s] for s in common], [qd[s] for s in common]).pvalue if stats else float("nan")
        line = (f"{ag}@{best[ag][1]} vs best QRC ({qa}@{best[qa][1]}): {rel:+.1f}%  diff {d.mean():+.4f} "
                f"[{d.mean() - half:+.4f}, {d.mean() + half:+.4f}]  p={pv:.4g}  ET better on {int((d < 0).sum())}/{len(d)}")
        print("  " + line)
        summary.append(f"{task:<26} {line}")
print(f"\nSUMMARY {prefix} (negative % = QRC-ET has lower value error than the best QRC)")
for s in summary:
    print("  " + s)
