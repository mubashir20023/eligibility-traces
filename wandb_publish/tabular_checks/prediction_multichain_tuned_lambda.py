"""Tabular check (26 Oct 2026): prediction multi-chain with TD's lambda tuned over {0, .24, .5, .9}.
Result (fresh seeds 100-109): ET with count-based Theta step still has 19-31% lower RMSE at m = 32 and 128.
Run:  python prediction_multichain_tuned_lambda.py   (from this folder)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from prediction_multichain import run

tune, test = range(0, 5), range(100, 110)
for m in [8, 32, 128]:
    row = []
    for name, method, lams in [("TD", "td", [0.0, 0.24, 0.5, 0.9]), ("ET count", "et_count", [0.5, 0.9])]:
        best = min((np.mean([run(m, method, s, a, l) for s in tune]), a, l) for a in [0.01, 0.03, 0.1, 0.3, 1.0] for l in lams)
        r = [run(m, method, s, best[1], best[2]) for s in test]
        row.append(f"{name}: {np.mean(r):.3f}±{np.std(r) / np.sqrt(10):.3f} (a={best[1]}, lam={best[2]})")
    print(f"m={m:<4d}" + " | ".join(row), flush=True)
