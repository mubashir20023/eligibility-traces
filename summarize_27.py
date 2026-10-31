"""R27: is Algo 3 at its best learning rate (q_lr 2e-3) better than QRC on FRESH seeds?  Run from the repo root: python summarize_27.py
Part 1 Freeway: seeds 1,2,3,42,999 (where q_lr was chosen) and seeds 4..13 (fresh, unbiased) separately, then all 15.
Part 2 Asterix at q_lr 2e-3, seeds 1,2,3,42,999.  Score = mean of the last 10% of training."""
import csv, glob, os
import numpy as np
from scipy import stats
root = "logs"
if "SCRATCH" in os.environ and os.path.isdir(os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")):
    root = os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")
S5 = [1, 2, 3, 42, 999]; NEW = list(range(4, 14)); BASE = {"seed1": 1, "seed2": 2, "seed3": 3, "seed4": 42, "seed5": 999}
last10 = lambda x: float(np.mean(x[-max(1, len(x) // 10):]))
def logs(env, exps):
    out = {}
    for e in exps:
        for f in glob.glob(f"{root}/**/{env}__{e}__*/training_data.csv", recursive=True):
            rows = list(csv.DictReader(open(f)))
            if len(rows) >= 4000: out[int(os.path.basename(os.path.dirname(f)).split("__")[-1])] = last10([float(r["avg_return"]) for r in rows])
    return out
def cmp(x, y, seeds):
    c = [s for s in seeds if s in x and s in y]
    if len(c) < 3: return None
    a = np.array([x[s] for s in c]); b = np.array([y[s] for s in c]); d = a - b
    return dict(n=len(c), a=a.mean(), b=b.mean(), d=d.mean(), ci=stats.t.ppf(.975, len(c) - 1) * d.std(ddof=1) / np.sqrt(len(c)), p=stats.ttest_rel(a, b).pvalue, worse=int((d < 0).sum()), dead=(int((b < 1).sum()), int((a < 1).sum())))
def show(label, r):
    print(f"  {label:<58}" + ("(not enough matched seeds yet)" if r is None else f"n={r['n']:<3}{r['a']:6.2f} vs {r['b']:6.2f}  diff {r['d']:+6.2f} (±{r['ci']:.2f})  p={r['p']:.3g}  first worse on {r['worse']}/{r['n']}  dead {r['dead'][1]}/{r['dead'][0]}"))
rows = list(csv.DictReader(open("plot_data/minatar_qrc_Freeway-v1.csv"))); base = {BASE[s]: last10([float(r[s]) for r in rows]) for s in BASE}
fw = "Freeway-v1"
alg = {**logs(fw, ["R26_algo3_lr2e3"]), **logs(fw, ["R27_algo3_lr2e3_more"])}
q2 = {**logs(fw, ["R26_qrc_lr2e3"]), **logs(fw, ["R27_qrc_lr2e3_more"])}
q5 = {**logs(fw, ["R24_qrc_lr5e4"]), **logs(fw, ["R26_qrc_lr5e4_more"])}
q1 = {**base, **logs(fw, ["R18_qrc"])}
print("Part 1. Freeway: Algo 3 (eta 0.3, q_lr 2e-3) minus QRC          (first number = Algo 3)")
for nm, seeds in (("seeds 1,2,3,42,999 (lr chosen here)", S5), ("seeds 4..13 (FRESH, unbiased)", NEW), ("all 15 seeds", S5 + NEW)):
    print(f" {nm}")
    show("vs QRC at the same lr 2e-3", cmp(alg, q2, seeds)); show("vs QRC at its best earlier lr 5e-4", cmp(alg, q5, seeds)); show("vs QRC at the default lr 1e-4", cmp(alg, q1, seeds))
print("\nPart 2. Asterix, q_lr 2e-3, seeds 1,2,3,42,999")
show("Algo 3 vs QRC", cmp(logs("Asterix-v1", ["R27_algo3_lr2e3"]), logs("Asterix-v1", ["R27_qrc_lr2e3"]), S5))
