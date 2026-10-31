"""Tabular sanity check (26 Oct 2026): Watkins Q(lambda) vs tabular QET(lambda, eta) on the multi-chain
CONTROL task with the only decision at the start state (multichain_env.MultiChainEnv).
Result: ET never beat Q(lambda). Metric: % of episodes in which the greedy start action is optimal.
Run from the repo root (needs multichain_env.py there):  python experiments/tabular_checks/control_start_decision.py
"""
import os, sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from multichain_env import MultiChainEnv


def run(m, sigma, eta, alpha, seed, lam=0.8, gamma=0.99, eps=0.1, episodes=1500, n=8):
    env = MultiChainEnv(m=m, n=n, sigma=sigma)
    rng = np.random.default_rng(seed + 1000)
    S, A = env.n_states, 2
    Q = np.zeros((S, A))
    Z = np.zeros((S, A, S, A))            # z(s,a): expected trace over (s',a') entries
    cnt = np.zeros((S, A))
    correct = []
    for ep in range(episodes):
        env.reset(seed=seed * 100000 + ep)
        s = env.state
        e = np.zeros((S, A)); y = np.zeros((S, A))
        correct.append(Q[0].argmax() == 0 and Q[0, 0] != Q[0, 1])
        done = False
        while not done:
            greedy = int(np.argmax(Q[s] + 1e-9 * rng.random(A)))
            a = int(rng.integers(A)) if rng.random() < eps else greedy
            _, r, done, _, _ = env.step(a)
            s2 = env.state
            delta = r + (0.0 if done else gamma * Q[s2].max()) - Q[s, a]
            x = np.zeros((S, A)); x[s, a] = 1.0
            e = gamma * lam * e + x
            cnt[s, a] += 1
            Z[s, a] += (e - Z[s, a]) / cnt[s, a]
            y = (1 - eta) * Z[s, a] + eta * (gamma * lam * y + x)
            Q += alpha * delta * y
            if a != greedy:
                e[:] = 0; y[:] = 0
            s = s2
    return 100 * np.mean(correct)


if __name__ == "__main__":
    seeds = range(8)
    for sigma in [0.0, 1.0]:
        for m in [1, 4, 16]:
            out = []
            for name, eta in [("Q(lambda)", 1.0), ("QET eta=.3", 0.3), ("QET eta=0", 0.0)]:
                best = max((np.mean([run(m, sigma, eta, a, s) for s in seeds]), a) for a in [0.1, 0.3, 0.5])
                out.append(f"{name}: {best[0]:5.1f}% (a={best[1]})")
            print(f"sigma={sigma} m={m:<3d} " + " | ".join(out), flush=True)
