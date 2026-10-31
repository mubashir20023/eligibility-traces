"""Does Algo 3's gap to QRC depend on environment stochasticity?  Run from the repo root:  python summarize_stoch.py
For each game and sticky-action probability p in {0.0, 0.1 (MinAtar default), 0.5}: per-seed score = mean of the last 10%
of training, restricted to the 8 seeds {1,2,3,42,999,4,5,6} so every p is compared on the same seeds. Reports QRC mean, the median difference, the number of dead runs (final score < 1) for each agent,
Algo3 mean, and the paired difference (Algo3 - QRC) with a 95% CI and paired p-value."""
import csv, glob, os
import numpy as np
from scipy import stats

root = "logs"
if "SCRATCH" in os.environ and os.path.isdir(os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")):
    root = os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")
SEEDS = [1, 2, 3, 42, 999, 4, 5, 6]
BASE = {"seed1": 1, "seed2": 2, "seed3": 3, "seed4": 42, "seed5": 999}
last10 = lambda x: float(np.mean(x[-max(1, len(x) // 10):]))

def logs(env, exps):
    out = {}
    for e in exps:
        for f in glob.glob(f"{root}/**/{env}__{e}__*/training_data.csv", recursive=True):
            rows = list(csv.DictReader(open(f)))
            if len(rows) >= 4000:
                out[int(os.path.basename(os.path.dirname(f)).split("__")[-1])] = last10([float(r["avg_return"]) for r in rows])
    return out

def baseline(env):
    f = f"plot_data/minatar_qrc_{env}.csv"
    if not os.path.exists(f): return {}
    rows = list(csv.DictReader(open(f)))
    return {BASE[s]: last10([float(r[s]) for r in rows]) for s in BASE if s in rows[0]}

print(f"{'game':<12}{'sticky p':>9}{'n':>4}{'QRC':>9}{'Algo3':>9}{'diff':>8}{'95% CI':>9}{'p':>9}{'worse':>8}{'median diff':>12}{'dead QRC/Algo3':>16}")
DIFFS = {}
for env in ("Freeway-v1", "Asterix-v1"):
    cfg = {
        0.0: (logs(env, ["R19_qrc_p00"]), logs(env, ["R19_algo3_p00"])),
        0.1: ({**baseline(env), **logs(env, ["R18_qrc"])}, logs(env, ["R17_algo3_paired", "R18_algo3_paired"])),
        0.5: (logs(env, ["R19_qrc_p50"]), logs(env, ["R19_algo3_p50"])),
    }
    for p, (q, a) in cfg.items():
        c = [s for s in SEEDS if s in q and s in a]
        if len(c) < 3:
            print(f"{env:<12}{p:>9}{len(c):>4}   (not enough matched seeds yet)"); continue
        qa = np.array([q[s] for s in c]); aa = np.array([a[s] for s in c]); d = aa - qa
        DIFFS[(env, p)] = dict(zip(c, d))
        h = stats.t.ppf(0.975, len(c) - 1) * d.std(ddof=1) / np.sqrt(len(c)); pv = stats.ttest_rel(aa, qa)[1]
        dq, da = int((qa < 1.0).sum()), int((aa < 1.0).sum())      # "dead" runs: final score below 1 (network never learned)
        print(f"{env:<12}{p:>9}{len(c):>4}{qa.mean():>9.2f}{aa.mean():>9.2f}{d.mean():>+8.2f}{'±%.2f' % h:>9}{pv:>9.3g}{int((d < 0).sum()):>5}/{len(c)}{np.median(d):>+12.2f}{f'{dq}/{da}':>16}")

print("\nDoes the Algo3-QRC gap change with stochasticity? (paired t-test on per-seed gap differences, seeds in both)")
for env in ("Freeway-v1", "Asterix-v1"):
    for p1, p2 in ((0.0, 0.1), (0.1, 0.5), (0.0, 0.5)):
        if (env, p1) in DIFFS and (env, p2) in DIFFS:
            c = sorted(set(DIFFS[(env, p1)]) & set(DIFFS[(env, p2)]))
            if len(c) >= 3:
                x = np.array([DIFFS[(env, p2)][s] - DIFFS[(env, p1)][s] for s in c])
                print(f"{env:<12} gap(p={p2}) - gap(p={p1}) = {x.mean():+.2f}  (95% CI +-{stats.t.ppf(0.975, len(c)-1) * x.std(ddof=1) / np.sqrt(len(c)):.2f}, p={stats.ttest_1samp(x, 0).pvalue:.3g}, n={len(c)})")
