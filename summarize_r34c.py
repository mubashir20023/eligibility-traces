"""Summarize R34c (MinAtar with the QRC-ET fixes). Score = mean rolling-100 return over the last 10% of 5M steps.
Per game: agent x learning-rate table, then best QRC-ET vs best QRC (paired over seeds 201-205).
Usage (from stream-rep-rl):  python summarize_r34c.py   |   python summarize_r34c.py R34c R35  (several jobs pooled)
"""
import csv, glob, math, re, sys
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

TOTAL = 5_000_000
PRE = "(?:" + "|".join(sys.argv[1:] or ["R34c"]) + ")"
score, div = defaultdict(dict), defaultdict(int)
for p in glob.glob("logs/**/*-v1__*/training_data.csv", recursive=True):
    m = re.search(rf"(\w+)-v1__{PRE}_(.+)_lr([\de.-]+)__(\d+)/", p)
    rows = list(csv.DictReader(open(p))) if m else []
    if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
        continue
    st = np.array([float(r["global_step"]) for r in rows]); ret = np.array([float(r["avg_return"]) for r in rows])
    q = np.array([float(r["q_values"]) for r in rows])
    key = (m.group(1), m.group(2), m.group(3))
    score[key][int(m.group(4))] = float(ret[st > 0.9 * TOTAL].mean())
    div[key] += int(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4)

for game in sorted({k[0] for k in score}):
    agents = sorted({k[1] for k in score if k[0] == game}, key=lambda a: (not a.startswith("qrc"), a))
    lrs = sorted({k[2] for k in score if k[0] == game}, key=float)
    print(f"\n=== {game}   (final score, mean ± s.e. [n, diverged]; higher = better)")
    print(f"{'agent':<14}" + "".join(f"{'lr ' + lr:>24}" for lr in lrs))
    best = {}
    for ag in agents:
        cells = []
        for lr in lrs:
            d = score.get((game, ag, lr), {})
            if d:
                v = np.array(list(d.values()))
                cells.append(f"{v.mean():.2f}±{v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.2f} [{len(v)},{div[(game, ag, lr)]}]")
                if ag not in best or v.mean() > best[ag][0]:
                    best[ag] = (v.mean(), lr)
            else:
                cells.append("-")
        print(f"{ag:<14}" + "".join(f"{c:>24}" for c in cells))
    q = [(best[a][0], a) for a in agents if a.startswith("qrc") and a in best]
    if not q:
        continue
    _, qa = max(q)
    for ag in agents:
        if ag.startswith("qrc") or ag not in best:
            continue
        e, b = score[(game, ag, best[ag][1])], score[(game, qa, best[qa][1])]
        common = sorted(set(e) & set(b))
        if len(common) < 2:
            continue
        d = np.array([e[s] - b[s] for s in common])
        half = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.78) * d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {ag}@{best[ag][1]} - best QRC ({qa}@{best[qa][1]}): {d.mean():+.2f} [{d.mean() - half:+.2f}, {d.mean() + half:+.2f}], "
              f"ET higher on {int((d > 0).sum())}/{len(d)}")
print("\nFive seeds per cell: treat as a screen.")
