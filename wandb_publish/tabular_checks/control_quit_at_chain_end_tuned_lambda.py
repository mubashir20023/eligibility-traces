import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""Tabular check (26 Oct 2026): same task, Q(lambda) with lambda tuned in {0,.24,.5,.8,.95}: ties ET in 5 of 6 settings."""
import numpy as np
from control_quit_at_chain_end import run
for sigma in [1.0, 2.0]:
    for m in [1, 8, 32]:
        res = {}
        for name, eta, lams in [("Q best lam", 1.0, [0.0, 0.24, 0.5, 0.8, 0.95]), ("ET0 best lam", 0.0, [0.5, 0.8, 0.95])]:
            res[name] = max((np.mean([run(m, sigma, eta, a, s, lam=l) for s in range(8)]), a, l)
                            for a in [0.03, 0.1, 0.3] for l in lams)
        print(f"sigma={sigma} m={m:<3d} " + " | ".join(f"{k}: {v[0]:.3f} (a={v[1]}, lam={v[2]})" for k, v in res.items()), flush=True)
