"""Tabular TD(lambda) vs tabular expected traces on MultiChainPred (same task as the deep R30 runs).

Three learners, all with constant step size alpha, expected trace z(s) = running mean of past traces (1/n step):
  TD        w += a * delta * e                        (ordinary trace)
  ET-now    w += a * delta * z(S_t)                   (paper Algorithm 1: z of the CURRENT state)
  ET-lag    w += a * delta * (g*l*z(S_{t-1}) + x_t)   (Algorithm-3 timing, eta = 0: z reaches the update one step late)
Saves value snapshots and credit maps (what one update at the bottleneck changes) for plotting, and prints RMSE.
Usage: python tabular_snapshots.py --m 32 --sigma 1.0 --out tabular_m32_s1.npz
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.abspath("."))
from multichain_env import MultiChainPredEnv


def run(method, m, sigma, alpha, lam, seed, steps, every, gamma=0.99):
    env = MultiChainPredEnv(m=m, n=4, sigma=sigma, gamma=gamma)
    env.reset(seed=seed)
    S, true = env.n_states, env.true_values()
    w, Z, cnt = np.zeros(S), np.zeros((S, S)), np.zeros(S)
    e, z_prev = np.zeros(S), np.zeros(S)
    snaps, s, t = [], env.state, 0
    while t < steps:
        _, r, done, _, _ = env.step(0)
        s2 = env.state
        x = np.zeros(S); x[s] = 1
        delta = r + (0.0 if done else gamma * w[s2]) - w[s]
        e = gamma * lam * e + x
        cnt[s] += 1
        Z[:, s] += (e - Z[:, s]) / cnt[s]
        u = {"TD": e, "ET-now": Z[:, s], "ET-lag": gamma * lam * z_prev + x}[method]
        w += alpha * delta * u
        z_prev = Z[:, s].copy()
        t += 1
        if done:
            env.reset(); e[:] = 0; z_prev[:] = 0
        s = env.state
        if t % every == 0:
            b = S - 1
            path = [0] + [1 + i for i in range(4)] + [b]
            et = np.zeros(S)
            for p in path:
                et = gamma * lam * et + np.eye(S)[p]
            credit = {"TD": et, "ET-now": Z[:, b], "ET-lag": gamma * lam * Z[:, path[-2]] + np.eye(S)[b]}[method]
            snaps.append((t, w.copy(), credit))
    return snaps, true


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=100000); ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = {}
    for method in ["TD", "ET-now", "ET-lag"]:
        best = None
        for alpha in [0.03, 0.1, 0.3]:
            for lam in ([0.0, 0.5, 0.9] if method == "TD" else [0.9]):
                errs = []
                for seed in range(5):
                    snaps, true = run(method, a.m, a.sigma, alpha, lam, seed, a.steps, a.every)
                    errs.append(np.mean([np.sqrt(np.mean((w - true) ** 2)) for _, w, _ in snaps]))
                if best is None or np.mean(errs) < best[0]:
                    best = (np.mean(errs), alpha, lam, np.std(errs) / np.sqrt(5))
        snaps, true = run(method, a.m, a.sigma, best[1], best[2], 0, a.save_steps, a.save_every)
        res[method] = dict(auc=best[0], sem=best[3], alpha=best[1], lam=best[2], snaps=snaps, true=true)
        print(f"m={a.m} sigma={a.sigma} {method:7s} RMSE-AUC {best[0]:.4f} ± {best[3]:.4f}  (alpha={best[1]}, lambda={best[2]})", flush=True)
    if a.out:
        np.savez_compressed(a.out, **{f"{k}_{f}": np.array([s[i] for s in v["snaps"]])
                                      for k, v in res.items() for i, f in enumerate(["step", "values", "credit"])},
                            true=res["TD"]["true"], chains=a.m, chain_len=4,
                            meta=str({k: {kk: v[kk] for kk in ("auc", "alpha", "lam")} for k, v in res.items()}))
