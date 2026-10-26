"""Summarize R28: final score = mean avg_return over the last 10% of steps, per seed; paired tests vs arm 1.

Usage (from the stream-rep-rl folder):  python summarize_r28.py
"""
import csv, glob, os, re, sys
import numpy as np
try:
    from scipy import stats
except ImportError:                                   # scipy missing: CI only, no p-value
    stats = None
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14}

ARMS = ["R28_et_eta03_lr2e3", "R28_zero_eta03_lr2e3", "R28_qrc_lam08_lr5e4",
        "R28_qrc_lam024_lr5e4", "R28_qrc_lam024_lr1e3"]
roots = ["logs"] + ([os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")] if "SCRATCH" in os.environ else [])
TOTAL = 5_000_000


def final_score(path):
    steps, rets = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(float(row["global_step"])); rets.append(float(row["avg_return"]))
    steps, rets = np.array(steps), np.array(rets)
    if len(steps) == 0 or steps.max() < 0.99 * TOTAL:
        return None                                   # unfinished run
    return rets[steps > 0.9 * TOTAL].mean()


scores = {}
for arm in ARMS:
    scores[arm] = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", f"*__{arm}__*", "training_data.csv"), recursive=True):
            seed = int(re.search(rf"__{arm}__(\d+)", p).group(1))
            s = final_score(p)
            if s is not None:
                scores[arm][seed] = s

print(f"{'arm':<24}{'n':>3}{'mean':>9}{'sem':>8}")
for arm in ARMS:
    v = np.array(list(scores[arm].values()))
    if len(v):
        print(f"{arm:<24}{len(v):>3}{v.mean():>9.2f}{v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:>8.2f}")
    else:
        print(f"{arm:<24}  0   (no finished runs yet)")

ref = ARMS[0]
print(f"\nPaired comparisons: {ref} minus other arm (same seeds only)")
for arm in ARMS[1:]:
    common = sorted(set(scores[ref]) & set(scores[arm]))
    if len(common) < 2:
        print(f"  vs {arm:<24} not enough paired seeds ({len(common)})"); continue
    d = np.array([scores[ref][s] - scores[arm][s] for s in common])
    tcrit = stats.t.ppf(0.975, len(d) - 1) if stats else T975.get(len(d) - 1, 2.0)
    half = tcrit * d.std(ddof=1) / np.sqrt(len(d))
    pval = stats.ttest_rel([scores[ref][s] for s in common], [scores[arm][s] for s in common]).pvalue if stats else float("nan")
    print(f"  vs {arm:<24} n={len(d):2d}  diff={d.mean():+6.2f}  95% CI [{d.mean() - half:+.2f}, {d.mean() + half:+.2f}]"
          f"  p={pval:.4f}  ET better on {int((d > 0).sum())}/{len(d)} seeds")
missing = {a: sorted(set(range(101, 116)) - set(scores[a])) for a in ARMS}
if any(missing.values()):
    print("\nUnfinished / missing seeds:", {a: m for a, m in missing.items() if m})

