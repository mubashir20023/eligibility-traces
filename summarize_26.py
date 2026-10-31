"""R26: Freeway learning-rate curves for QRC and Algo 3, and the 15-seed best-vs-best comparison.  Run from the repo root: python summarize_26.py
Score = mean of the last 10% of training. Part 1 uses seeds 1,2,3,42,999 (paired). Part 2: QRC at q_lr 5e-4 vs Algo 3 (eta 0.3) at q_lr 1e-3 on 15 seeds."""
import csv, glob, os
import numpy as np
from scipy import stats
root = "logs"
if "SCRATCH" in os.environ and os.path.isdir(os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")):
    root = os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")
S5 = [1, 2, 3, 42, 999]; S15 = S5 + list(range(4, 14)); BASE = {"seed1": 1, "seed2": 2, "seed3": 3, "seed4": 42, "seed5": 999}
last10 = lambda x: float(np.mean(x[-max(1, len(x) // 10):]))
def logs(exps):
    out = {}
    for e in exps:
        for f in glob.glob(f"{root}/**/Freeway-v1__{e}__*/training_data.csv", recursive=True):
            rows = list(csv.DictReader(open(f)))
            if len(rows) >= 4000: out[int(os.path.basename(os.path.dirname(f)).split("__")[-1])] = last10([float(r["avg_return"]) for r in rows])
    return out
rows = list(csv.DictReader(open("plot_data/minatar_qrc_Freeway-v1.csv"))); base = {BASE[s]: last10([float(r[s]) for r in rows]) for s in BASE}
QRC = {"1e-4": base, "2e-4": logs(["R24_qrc_lr2e4"]), "5e-4": logs(["R24_qrc_lr5e4"]), "1e-3": logs(["R25_qrc_lr1e3"]), "2e-3": logs(["R26_qrc_lr2e3"])}
ALG = {"1e-4": logs(["R17_algo3_paired"]), "2e-4": logs(["R24_algo3_lr2e4"]), "5e-4": logs(["R24_algo3_lr5e4"]), "1e-3": logs(["R25_algo3_lr1e3"]),
       "2e-3": logs(["R26_algo3_lr2e3"]), "5e-3": logs(["R26_algo3_lr5e3"])}
print("Freeway, seeds 1,2,3,42,999: mean score by q_lr   (n in brackets)")
print(f"{'q_lr':>6}{'QRC':>14}{'Algo 3 (eta 0.3)':>20}")
for lr in ["1e-4", "2e-4", "5e-4", "1e-3", "2e-3", "5e-3"]:
    f = lambda d: (lambda v: f"{np.mean(v):6.2f} [{len(v)}]" if v else "    -    ")([d[s] for s in S5 if s in d]) if d else "    -    "
    print(f"{lr:>6}{f(QRC.get(lr, {})):>14}{f(ALG.get(lr, {})):>20}")
q = {**logs(["R24_qrc_lr5e4"]), **logs(["R26_qrc_lr5e4_more"])}; a = {**logs(["R25_algo3_lr1e3"]), **logs(["R26_algo3_lr1e3_more"])}
c = [s for s in S15 if s in q and s in a]
print(f"\nBest vs best, 15 seeds: QRC q_lr 5e-4 vs Algo 3 q_lr 1e-3   (matched seeds so far: {len(c)})")
if len(c) >= 3:
    x = np.array([a[s] for s in c]); y = np.array([q[s] for s in c]); d = x - y
    h = stats.t.ppf(.975, len(c) - 1) * d.std(ddof=1) / np.sqrt(len(c))
    print(f"  QRC {y.mean():.2f} ± {y.std(ddof=1):.2f} | Algo 3 {x.mean():.2f} ± {x.std(ddof=1):.2f} | Algo 3 - QRC = {d.mean():+.2f} (95% CI ±{h:.2f}), paired p={stats.ttest_rel(x, y).pvalue:.3g}, Algo 3 worse on {int((d < 0).sum())}/{len(c)}, dead QRC/Algo 3: {int((y < 1).sum())}/{int((x < 1).sum())}")
