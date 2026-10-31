"""15-seed comparison, plain QRC vs Algo 3 (eta 0.3, paired RNG), Freeway and Asterix.  Run from the repo root:
    python summarize_15.py
Score per seed = mean of the last 10% of training (avg_return). QRC seeds 1,2,3,42,999 come from the repo baseline
(plot_data; verified identical to R17_qrc_repro); new seeds from logs R18_qrc.  Algo 3 seeds from R17_algo3_paired
(Freeway) + R18_algo3_paired.  Paired statistics use the seeds both agents have."""
import csv, glob, os, sys
import numpy as np
from scipy import stats

root = "logs"
if "SCRATCH" in os.environ and os.path.isdir(os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")):
    root = os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")
BASE_SEEDS = {"seed1": 1, "seed2": 2, "seed3": 3, "seed4": 42, "seed5": 999}

def last10(x): return float(np.mean(x[-max(1, len(x) // 10):]))

def from_logs(env, exps):
    out = {}
    for exp in exps:
        for f in glob.glob(f"{root}/**/{env}__{exp}__*/training_data.csv", recursive=True):
            seed = int(os.path.basename(os.path.dirname(f)).split("__")[-1])
            rows = list(csv.DictReader(open(f)))
            if len(rows) >= 4000:
                out[seed] = last10([float(r["avg_return"]) for r in rows])
    return out

def baseline(env, algo):
    f = f"plot_data/minatar_{algo}_{env}.csv"
    if not os.path.exists(f): return {}
    rows = list(csv.DictReader(open(f)))
    return {BASE_SEEDS[s]: last10([float(r[s]) for r in rows]) for s in BASE_SEEDS if s in rows[0]}

def ci95(d):
    n = len(d); m = float(np.mean(d))
    if n < 2: return m, float("nan")
    h = stats.t.ppf(0.975, n - 1) * np.std(d, ddof=1) / np.sqrt(n)
    return m, float(h)

for env in ("Freeway-v1", "Asterix-v1"):
    qrc = {**baseline(env, "qrc"), **from_logs(env, ["R18_qrc"])}
    a3 = from_logs(env, ["R17_algo3_paired", "R18_algo3_paired"])
    print(f"\n===== {env} =====")
    seeds = sorted(set(qrc) | set(a3))
    print("seed   " + " ".join(f"{s:>6}" for s in seeds))
    print("QRC    " + " ".join(f"{qrc[s]:6.1f}" if s in qrc else "     -" for s in seeds))
    print("Algo3  " + " ".join(f"{a3[s]:6.1f}" if s in a3 else "     -" for s in seeds))
    for name, d in (("QRC", qrc), ("Algo3 eta0.3", a3)):
        if d:
            m, h = ci95(list(d.values()))
            print(f"{name:<13} n={len(d):<3} mean={m:6.2f}  sd={np.std(list(d.values()), ddof=1) if len(d)>1 else 0:5.2f}  95%CI=±{h:.2f}")
    common = sorted(set(qrc) & set(a3))
    if len(common) >= 3:
        diff = np.array([a3[s] - qrc[s] for s in common]); m, h = ci95(diff)
        t, p = stats.ttest_rel([a3[s] for s in common], [qrc[s] for s in common])
        print(f"paired (n={len(common)} matched seeds): Algo3 - QRC = {m:+.2f} ± {h:.2f} (95% CI), paired t-test p={p:.4g}, seeds where Algo3 is worse: {int(np.sum(diff<0))}/{len(common)}")
