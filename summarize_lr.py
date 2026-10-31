"""Does Algo 3 do better than QRC at a larger q_lr?  Run from the repo root:  python summarize_lr.py
Seeds 1,2,3,42,999 only. q_lr 1e-4 reference = existing runs (repo baseline QRC; R17/R18 Algo 3). Score = mean of the last 10%."""
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
def baseline(env):
    f = f"plot_data/minatar_qrc_{env}.csv"
    rows = list(csv.DictReader(open(f))); return {BASE[s]: last10([float(r[s]) for r in rows]) for s in BASE}
print(f"{'game':<12}{'q_lr':>8}{'n':>3}{'QRC':>8}{'Algo3':>8}{'diff':>8}{'95% CI':>9}{'p':>8}{'worse':>7}{'dead QRC/Algo3':>16}")
for env in ("Freeway-v1", "Asterix-v1"):
    cfg = {"1e-4": (baseline(env), logs(env, ["R17_algo3_paired", "R18_algo3_paired"])),
           "2e-4": (logs(env, ["R24_qrc_lr2e4"]), logs(env, ["R24_algo3_lr2e4"])),
           "5e-4": (logs(env, ["R24_qrc_lr5e4"]), logs(env, ["R24_algo3_lr5e4"]))}
    for lr, (q, a) in cfg.items():
        c = [s for s in SEEDS if s in q and s in a]
        if len(c) < 3: print(f"{env:<12}{lr:>8}{len(c):>3}   (not enough matched seeds yet)"); continue
        qa = np.array([q[s] for s in c]); aa = np.array([a[s] for s in c]); d = aa - qa
        h = stats.t.ppf(0.975, len(c) - 1) * d.std(ddof=1) / np.sqrt(len(c)); pv = stats.ttest_rel(aa, qa)[1]
        print(f"{env:<12}{lr:>8}{len(c):>3}{qa.mean():>8.2f}{aa.mean():>8.2f}{d.mean():>+8.2f}{'±%.2f' % h:>9}{pv:>8.3g}{int((d < 0).sum()):>4}/{len(c)}{f'{int((qa < 1).sum())}/{int((aa < 1).sum())}':>16}")
