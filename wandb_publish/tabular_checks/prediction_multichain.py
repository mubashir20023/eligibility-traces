"""Rung 1 of the validation ladder: tabular multi-chain prediction (van Hasselt et al. 2021, Fig. 3/4).

Compares, on the same seeds and with the step size / lambda tuned per method:
  TD(lambda)            eta = 1                         (standard trace)
  damping control       y = eta * (lambda*y + x), z = 0  (== TD(lambda*eta) with step scaled by eta)
  ET, constant beta     Theta updated with a fixed step  (z starts at 0 -> shrunken for rare states)
  ET, count beta        Theta updated with 1/n(s)        (z is the exact running mean of past traces)

Environment: start -> uniformly one of m chains of length n -> shared bottleneck -> terminal.
Intermediate rewards 0; terminal reward +1 w.p. 0.9, -1 w.p. 0.1; gamma = 1, so v(s) = 0.8 everywhere.

Run: python3 multichain_et_reference.py      (~2-3 minutes)
"""
import numpy as np


def run(m, method, seed, alpha, lam, eta=0.0, beta=0.1, n=4, episodes=1024):
    rng = np.random.default_rng(seed)
    S = 1 + m * n + 1
    bottleneck = S - 1
    w = np.zeros(S)
    Theta = np.zeros((S, S))                         # z(s) = Theta @ x(s) = Theta[:, s]
    count = np.zeros(S)
    for _ in range(episodes):
        c = rng.integers(m)
        states = [0] + [1 + c * n + i for i in range(n)] + [bottleneck]
        rewards = [0.0] * (len(states) - 1) + [1.0 if rng.random() < 0.9 else -1.0]
        e = np.zeros(S)                              # instantaneous trace: the target for Theta
        y = np.zeros(S)                              # trace actually used in the w update
        for k, s in enumerate(states):
            count[s] += 1
            x = np.zeros(S); x[s] = 1.0
            v_next = w[states[k + 1]] if k + 1 < len(states) else 0.0
            delta = rewards[k] + v_next - w[s]
            e = lam * e + x
            if method == "td":
                y = e
            elif method == "damping":
                y = eta * (lam * y + x)              # Eq. 3 with z forced to 0
            else:
                step = 1.0 / count[s] if method == "et_count" else beta
                Theta[:, s] += step * (e - Theta[:, s])
                y = (1 - eta) * Theta[:, s] + eta * (lam * y + x)   # Eq. 3
            w += alpha * delta * y
    return np.sqrt(np.mean((w - 0.8) ** 2))


def tuned(m, method, seeds, **kw):
    grid = [(a, l) for a in [0.01, 0.03, 0.1, 0.3, 1.0] for l in [0.5, 0.9]]
    scores = [(np.mean([run(m, method, s, a, l, **kw) for s in seeds]), a, l) for a, l in grid]
    return min(scores)


if __name__ == "__main__":
    tune_seeds, test_seeds = range(0, 5), range(100, 110)
    methods = [("TD(lambda)", "td", {}),
               ("damping eta=.3", "damping", {"eta": 0.3}),
               ("ET const beta", "et_const", {}),
               ("ET count beta", "et_count", {})]
    print("RMSE after 1024 episodes; step size and lambda tuned on seeds 0-4, reported on fresh seeds 100-109")
    for m in [8, 32, 128]:
        row = []
        for name, method, kw in methods:
            _, a, l = tuned(m, method, tune_seeds, **kw)
            r = [run(m, method, s, a, l, **kw) for s in test_seeds]
            row.append(f"{name}: {np.mean(r):.3f}±{np.std(r) / np.sqrt(len(r)):.3f}")
        print(f"m={m:<4d} " + " | ".join(row))
