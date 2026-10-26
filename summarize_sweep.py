"""Learning-rate sweep table for R28 + R29 (Freeway, seeds 101-115).

Score per run = mean avg_return over the last 10% of steps. A run is marked DIVERGED if
q_values ever became nan/inf or |q| > 1e4. Usage (from stream-rep-rl):  python summarize_sweep.py
"""
import csv, math, os
import numpy as np

SEEDS = range(101, 116)
TOTAL = 5_000_000
GRID = {  # agent -> {lr: exp_name}
    "QRC lam0.8":  {"5e-4": "R28_qrc_lam08_lr5e4", "1e-3": "R29_qrc_lam08_lr1e3", "2e-3": "R29_qrc_lam08_lr2e3"},
    "QRC lam0.24": {"5e-4": "R28_qrc_lam024_lr5e4", "1e-3": "R28_qrc_lam024_lr1e3", "2e-3": "R29_qrc_lam024_lr2e3"},
    "ET eta0.3":   {"5e-4": "R29_et_eta03_lr5e4", "1e-3": "R29_et_eta03_lr1e3", "2e-3": "R28_et_eta03_lr2e3",
                    "4e-3": "R29_et_eta03_lr4e3"},
    "ET Theta off": {"2e-3": "R28_zero_eta03_lr2e3"},
}
LRS = ["5e-4", "1e-3", "2e-3", "4e-3"]


def load(exp, seed):
    p = f"logs/MinAtar/Freeway-v1__{exp}__{seed}/training_data.csv"
    if not os.path.exists(p):
        return None
    steps, rets, diverged = [], [], False
    with open(p) as f:
        for r in csv.DictReader(f):
            steps.append(float(r["global_step"])); rets.append(float(r["avg_return"]))
            q = float(r["q_values"])
            if math.isnan(q) or math.isinf(q) or abs(q) > 1e4:
                diverged = True
    steps, rets = np.array(steps), np.array(rets)
    if len(steps) == 0 or steps.max() < 0.99 * TOTAL:
        return None
    return rets[steps > 0.9 * TOTAL].mean(), diverged


res = {}
print(f"{'agent':<14}" + "".join(f"{lr:>22}" for lr in LRS))
for agent, row in GRID.items():
    cells = []
    for lr in LRS:
        exp = row.get(lr)
        runs = {s: load(exp, s) for s in SEEDS} if exp else {}
        runs = {s: v for s, v in runs.items() if v is not None}
        res[(agent, lr)] = {s: v[0] for s, v in runs.items()}
        if not runs:
            cells.append("-" if not exp else "not finished"); continue
        v = np.array([x[0] for x in runs.values()]); div = sum(x[1] for x in runs.values())
        sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0
        cells.append(f"{v.mean():5.1f}±{sem:3.1f} n={len(v)} d={div}")
    print(f"{agent:<14}" + "".join(f"{c:>22}" for c in cells))
print("(d = number of runs that diverged)")


def best(agent):
    opts = [(np.mean(list(v.values())), lr) for (a, lr), v in res.items() if a == agent and len(v) == 15]
    return max(opts) if opts else None


print("\nBest learning rate per agent, then paired comparison of best ET vs best QRC (same seeds):")
b_et = best("ET eta0.3")
for q in ["QRC lam0.8", "QRC lam0.24"]:
    b_q = best(q)
    if not (b_et and b_q):
        print(f"  {q}: not enough finished runs"); continue
    e, o = res[("ET eta0.3", b_et[1])], res[(q, b_q[1])]
    d = np.array([e[s] - o[s] for s in SEEDS])
    half = 2.145 * d.std(ddof=1) / np.sqrt(len(d))           # t(0.975, 14)
    print(f"  ET@{b_et[1]} ({b_et[0]:.1f}) - {q}@{b_q[1]} ({b_q[0]:.1f}) = {d.mean():+.2f}, "
          f"95% CI [{d.mean() - half:+.2f}, {d.mean() + half:+.2f}], ET better on {int((d > 0).sum())}/15")
print("Note: 'best' is picked on these same seeds, so best-vs-best is slightly optimistic for both sides.")

