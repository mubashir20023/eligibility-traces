"""R25: (a) is Algo 3 just 'QRC with a shorter trace'?  (b) behaviour at q_lr 1e-3.   Run from the repo root: python summarize_25.py
Seeds 1,2,3,42,999 (paired). Score = mean of the last 10%. References: baseline QRC (lambda 0.8, q_lr 1e-4) and Algo 3 eta 0.3 (R17/R18)."""
import csv, glob, os
import numpy as np
from scipy import stats
root = "logs"
if "SCRATCH" in os.environ and os.path.isdir(os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")):
    root = os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")
SEEDS = [1, 2, 3, 42, 999]; BASE = {"seed1": 1, "seed2": 2, "seed3": 3, "seed4": 42, "seed5": 999}
last10 = lambda x: float(np.mean(x[-max(1, len(x) // 10):]))
def logs(env, exps):
    out = {}
    for e in exps:
        for f in glob.glob(f"{root}/**/{env}__{e}__*/training_data.csv", recursive=True):
            rows = list(csv.DictReader(open(f)))
            if len(rows) >= 4000: out[int(os.path.basename(os.path.dirname(f)).split("__")[-1])] = last10([float(r["avg_return"]) for r in rows])
    return out
def base(env):
    rows = list(csv.DictReader(open(f"plot_data/minatar_qrc_{env}.csv"))); return {BASE[s]: last10([float(r[s]) for r in rows]) for s in BASE}
def cmp(a, b):
    c = [s for s in SEEDS if s in a and s in b]
    if len(c) < 3: return None
    x = np.array([a[s] for s in c]); y = np.array([b[s] for s in c]); d = x - y
    return len(c), x.mean(), y.mean(), d.mean(), stats.t.ppf(.975, len(c) - 1) * d.std(ddof=1) / np.sqrt(len(c)), stats.ttest_rel(x, y)[1], int((x < y).sum())
def line(label, r):
    print(f"  {label:<46}" + ("(not enough matched seeds yet)" if r is None else f"n={r[0]}  {r[1]:6.2f} vs {r[2]:6.2f}  diff {r[3]:+7.2f} (±{r[4]:.2f}) p={r[5]:.3g}  first is worse on {r[6]}/{r[0]}"))
for env in ("Freeway-v1", "Asterix-v1"):
    q = base(env); a3 = logs(env, ["R17_algo3_paired", "R18_algo3_paired"])
    l3, l5 = logs(env, ["R25_qrc_lam03"]), logs(env, ["R25_qrc_lam05"])
    q13, a13 = logs(env, ["R25_qrc_lr1e3"]), logs(env, ["R25_algo3_lr1e3"])
    print(f"\n== {env}  (means over seeds present)")
    for nm, d in (("QRC lambda 0.8 (baseline)", q), ("QRC lambda 0.5", l5), ("QRC lambda 0.3", l3), ("Algo 3 eta 0.3 (lambda 0.8)", {s: a3[s] for s in a3 if s in SEEDS}),
                  ("QRC lr 1e-3", q13), ("Algo 3 lr 1e-3", a13)):
        v = [d[s] for s in SEEDS if s in d]; print(f"  {nm:<30} n={len(v)} mean={np.mean(v):6.2f}" if v else f"  {nm:<30} (none yet)")
    print(" Is Algo 3 just a shorter trace?  (Algo 3 minus QRC with shorter lambda; close to 0 = same behaviour)")
    line("Algo 3 vs QRC lambda 0.5", cmp(a3, l5)); line("Algo 3 vs QRC lambda 0.3", cmp(a3, l3))
    print(" Does a shorter lambda alone change QRC?")
    line("QRC lambda 0.5 vs QRC lambda 0.8", cmp(l5, q)); line("QRC lambda 0.3 vs QRC lambda 0.8", cmp(l3, q))
    print(" At q_lr 1e-3")
    line("Algo 3 vs QRC", cmp(a13, q13))
