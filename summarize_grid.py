"""Summarise gridworld_et.py results. Run from the repo root:  python summarize_grid.py [gridworld_out]
Per config folder and per eta: final |V - V*| (median, mean, sd, worst seed), number of seeds that blew up
(final V-error > 1.0, i.e. values far from the true ones), median recent steps-to-goal, mean Theta relative error.
MEDIAN is the number to trust when a few seeds diverge (the mean is then dominated by them)."""
import glob, json, os, sys
import numpy as np
root = sys.argv[1] if len(sys.argv) > 1 else "gridworld_out"
for f in sorted(glob.glob(os.path.join(root, "*", "results.json"))):
    d = json.load(open(f)); a = d["args"]
    print(f"\n== {os.path.basename(os.path.dirname(f))}: slip={a['slip']} hidden={a['hidden']} lr={a['lr']} steps={a['steps']} seeds={a['seeds']} reward_noise={a.get('reward_noise', 0)}")
    print(f"{'eta':>5} {'V-err median':>13} {'mean':>7} {'sd':>7} {'worst':>7} {'blew up':>8} {'steps-to-goal (median)':>23} {'Theta err':>10}")
    for eta, runs in d["results"].items():
        v = np.array([r["qerr"][-1] for r in runs], dtype=float); ln = np.array([r["lens"][-1] for r in runs]); th = np.array([r["theta_rel_err"] for r in runs], dtype=float)
        bad = ~np.isfinite(v) | (v > 1.0); ok = v[np.isfinite(v)]          # NaN / inf count as blow-ups
        if len(ok) == 0:
            print(f"{float(eta):>5} {'all NaN':>13} {'-':>7} {'-':>7} {'-':>7} {int(bad.sum()):>5}/{len(v):<2} {np.median(ln):>23.1f} {'-':>10}"); continue
        print(f"{float(eta):>5} {np.median(ok):>13.3f} {ok.mean():>7.3f} {ok.std(ddof=1) if len(ok) > 1 else 0:>7.3f} {ok.max():>7.3f} {int(bad.sum()):>5}/{len(v):<2} {np.median(ln):>23.1f} {np.nanmean(th[np.isfinite(th)]) if np.isfinite(th).any() else float('nan'):>10.3f}")
