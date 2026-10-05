"""Summarise R12 runs. Usage: python summarize_algo3.py [root_dir]
root_dir defaults to $SCRATCH/streamRepL/logs if SCRATCH set, else ./logs"""
import csv, glob, os, sys, collections
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else (
    os.path.join(os.environ["SCRATCH"], "streamRepL", "logs") if "SCRATCH" in os.environ else "logs")
BASE = {"Asterix": 19.91, "Breakout": 12.70, "Freeway": 53.41, "Seaquest": 42.07, "SpaceInvaders": 108.98}  # QRC
ET   = {"Asterix": 22.97, "Breakout": 15.64, "Freeway": 47.77, "Seaquest": 44.86, "SpaceInvaders": 97.60}  # QRC-ET
res = collections.defaultdict(list)
for f in sorted(glob.glob(f"{root}/**/*__R12_et_algo3__*/training_data.csv", recursive=True)):
    run = os.path.basename(os.path.dirname(f)); game = run.split("__")[0].split("/")[-1].split("-")[0]
    rows = list(csv.DictReader(open(f)))
    if len(rows) < 100:
        print(f"!! {run}: only {len(rows)} rows (incomplete / resumed-overwrite?)"); continue
    last = rows[-max(1, len(rows) // 10):]                       # final 10% of training
    ret = np.mean([float(r["avg_return"]) for r in last])
    upd = np.mean([float(r["q_update_l2"]) for r in last])
    dead = " <-- DEAD? (near-zero updates)" if upd < 1e-4 else ""
    print(f"{run:55s} return={ret:8.2f}  q_update_l2={upd:.2e}{dead}")
    res[game].append(ret)
print("\n%-14s %5s %14s %8s %8s" % ("game", "n", "algo3 mean±sd", "QRC", "QRC-ET"))
for g, v in res.items():
    print("%-14s %5d %8.2f±%-5.2f %8.2f %8.2f" % (g, len(v), np.mean(v), np.std(v), BASE.get(g, np.nan), ET.get(g, np.nan)))
