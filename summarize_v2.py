"""NOTE: the final-row metric is noisy (SpaceInvaders swings 75-117 within a run); prefer last10% for conclusions.
Compare finished runs against the repo's per-seed QRC / QRC-ET baselines.
Usage: python summarize_v2.py R12_et_algo3 R13_ctrl_eta1 R13_lr003   (run from repo root)
Uses the SAME metric as summary_results.csv: the last logged avg_return (final row),
and also prints the last-10% mean and the learning curve at 1M/2M/3M/5M."""
import csv, glob, os, sys, collections
import numpy as np
root = "logs"
if "SCRATCH" in os.environ and os.path.isdir(os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")):
    root = os.path.join(os.environ["SCRATCH"], "streamRepL", "logs")
STEPS = (1_000_000, 2_000_000, 3_000_000, 4_999_000)

def base(algo, env):
    f = f"plot_data/minatar_{algo}_{env}.csv"
    if not os.path.exists(f): return None
    rows = list(csv.DictReader(open(f)))
    return {s: [float(r[s]) for r in rows] for s in rows[0] if s.startswith("seed")}, [int(r["step"]) for r in rows]

def at(series, steps, s):
    return series[steps.index(min(steps, key=lambda x: abs(x - s)))]

print("=== BASELINES (repo plot_data): final value per seed, and value at 1M/2M/3M ===")
for env in ("Freeway-v1", "Seaquest-v1"):
    for algo in ("qrc", "qrc-et"):
        b = base(algo, env)
        if not b: continue
        d, st = b
        fin = [v[-1] for v in d.values()]
        l10b = [float(np.mean(v[-len(v)//10:])) for v in d.values()]
        print(f"{env:12s}{algo:7s} final mean={np.mean(fin):6.2f} sd={np.std(fin):5.2f} seeds={np.round(fin,1)}  | last10% mean={np.mean(l10b):6.2f}")
        print("   mean curve at 1M/2M/3M:", [round(float(np.mean([at(v, st, s) for v in d.values()])), 1) for s in STEPS[:3]])

for exp in sys.argv[1:]:
    print(f"\n=== {exp} ===")
    per = collections.defaultdict(list); l10 = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{root}/**/*__{exp}__*/training_data.csv", recursive=True)):
        run = os.path.basename(os.path.dirname(f)); env = run.split("__")[0]; seed = run.split("__")[-1]
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 4000:
            print(f"!! {run}: only {len(rows)} rows (incomplete)"); continue
        r = [float(x["avg_return"]) for x in rows]; steps = [int(x["global_step"]) for x in rows]
        fin = r[-1]; l10v = float(np.mean(r[-len(r)//10:]))
        upd = float(np.mean([float(x["q_update_l2"]) for x in rows[-len(rows)//10:]]))
        curve = [round(at(r, steps, s), 1) for s in STEPS]
        print(f"{run:50s} final={fin:7.2f} last10%={l10v:7.2f} q_upd={upd:.2e}  curve@1M/2M/3M/5M={curve}")
        per[env].append(fin); l10[env].append(l10v)
    for env, v in per.items():
        print(f"  -> {env}: final mean={np.mean(v):.2f} sd={np.std(v):.2f} | last10% mean={np.mean(l10[env]):.2f} sd={np.std(l10[env]):.2f} (n={len(v)})")
