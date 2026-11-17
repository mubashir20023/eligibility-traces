"""Re-analysis of the multi-chain runs (no new runs needed): early vs late learning, and same-lambda comparisons.

The expected-traces paper reports the value error after 1,024 episodes (early learning) and compares TD(lambda) with
ET(lambda) at each lambda. Our summaries used the mean error over the whole 120k-step run (about 20,000 episodes) and
compared QRC-ET (lambda 0.9) with the best QRC over lambda in {0, 0.9}. This script reports, for four windows
    after ~1,024 episodes (steps 5k-7k) | first 10% of the run | whole run | last 20% of the run
the best learning rate of each agent family in that window, and QRC-ET against
    QRC lambda 0.9 (same trace length),  QRC lambda 0,  and the best of the two.
Negative % = QRC-ET has the lower value error. Paired over seeds; * = 95% interval excludes 0.

Usage (from stream-rep-rl):  python summarize_windows.py R40 R40b R40c        (feature-sharing codes)
                             python summarize_windows.py R34a R35a            (one-hot, 20 fresh seeds)
"""
import csv, glob, re, sys
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

PREFIXES = sys.argv[1:] or ["R40", "R40b", "R40c"]
PRE = "(?:" + "|".join(PREFIXES) + ")"
WINDOWS = [("after ~1,024 episodes", 5000, 7000), ("first 10% of run", 0, 12000), ("whole run", 0, 10 ** 9), ("last 20% of run", 96000, 10 ** 9)]


def family(agent):
    if agent == "qrc_lam0":
        return "QRC lam 0"
    if agent == "qrc_lam09":
        return "QRC lam 0.9"
    if agent.startswith("et_oracle"):
        return "ET, theta on state id"
    if agent.startswith("et_split") or agent == "et_nolag_b01":
        return "ET"
    return None                                   # lagged / other diagnostic arms are left out


data = defaultdict(dict)                          # (task, family, agent, lr) -> seed -> (steps, rmse)
for p in glob.glob("logs/multichain_env:MultiChain*/training_data.csv"):
    m = re.search(rf"multichain_env:MultiChain(?:Code|CodeId|Sparse)-(m\d+-n\d+-p[\d.]+(?:-D\d+-k\d+)?)-v0__{PRE}_(.+)_lr([\d.e-]+)__(\d+)/", p)
    if not m or family(m.group(2)) is None:
        continue
    rows = list(csv.DictReader(open(p)))
    if not rows or "value_rmse" not in rows[0] or float(rows[-1]["global_step"]) < 0.99 * 120000:
        continue
    st = np.array([float(r["global_step"]) for r in rows])
    v = np.array([float(r["value_rmse"]) for r in rows]); v = np.where(np.isfinite(v), v, 10.0)
    data[(m.group(1), family(m.group(2)), m.group(2), m.group(3))][int(m.group(4))] = (st, v)


def pair(e, q):
    common = sorted(set(e) & set(q))
    if len(common) < 2:
        return "       -"
    d = np.array([e[s] - q[s] for s in common]); qm = np.mean([q[s] for s in common])
    h = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.26) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{100 * d.mean() / qm:+6.1f}%{'*' if (d.mean() + h < 0 or d.mean() - h > 0) else ' '}"


def order(t):
    D = re.search(r"-D(\d+)", t)
    return (t.split("-D")[0], -(int(D.group(1)) if D else 10 ** 6))


summary = defaultdict(list)
for task in sorted({k[0] for k in data}, key=order):
    print(f"\n=== {task}")
    for wname, lo, hi in WINDOWS:
        best = {}
        for (t, fam, ag, lr), runs in data.items():
            if t != task:
                continue
            vals = {s: float(v[(st >= lo) & (st <= hi)].mean()) for s, (st, v) in runs.items() if ((st >= lo) & (st <= hi)).any()}
            if len(vals) >= 2 and (fam not in best or np.mean(list(vals.values())) < best[fam][0]):
                best[fam] = (float(np.mean(list(vals.values()))), f"{ag}@{lr}", vals)
        q_all = [best[f] for f in ("QRC lam 0", "QRC lam 0.9") if f in best]
        if not q_all:
            continue
        qb = min(q_all, key=lambda b: b[0])
        line = f"  {wname:22s} " + "  ".join(f"{f}: {best[f][0]:.4f} ({best[f][1].split('@')[1]})" for f in ("QRC lam 0", "QRC lam 0.9", "ET", "ET, theta on state id") if f in best)
        print(line)
        for fam in ("ET", "ET, theta on state id"):
            if fam in best:
                e = best[fam][2]
                c9 = pair(e, best["QRC lam 0.9"][2]) if "QRC lam 0.9" in best else "       -"
                c0 = pair(e, best["QRC lam 0"][2]) if "QRC lam 0" in best else "       -"
                cb = pair(e, qb[2])
                print(f"  {'':22s}   {fam:22s} vs QRC lam 0.9: {c9}   vs QRC lam 0: {c0}   vs best QRC: {cb}")
                summary[(wname, fam)].append((task, c9, c0, cb))
print(f"\nSUMMARY {'+'.join(PREFIXES)}  (QRC-ET value error relative to QRC; negative = QRC-ET better; * = significant)")
for (wname, fam), rows in summary.items():
    print(f"\n  {wname} | {fam}")
    print(f"    {'task':24s} {'vs QRC lam 0.9':>15s} {'vs QRC lam 0':>14s} {'vs best QRC':>13s}")
    for task, c9, c0, cb in rows:
        print(f"    {task:24s} {c9:>15s} {c0:>14s} {cb:>13s}")
