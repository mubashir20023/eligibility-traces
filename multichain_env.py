"""Multi-chain CONTROL task for testing expected traces with the unchanged QRC / QRC-ET agents.

Control version of the multi-chain in van Hasselt et al. (2021), Fig. 3:

    start --action 0--> one of m chains of group A (chosen at random) --> bottleneck A --> terminal
          --action 1--> one of m chains of group B (chosen at random) --> bottleneck B --> terminal

* Only the action at the start state matters; inside chains and at the bottlenecks every action
  just moves forward. All rewards are 0 except the last one: mean 1.0 for group A, 0.6 for group B,
  plus Gaussian noise with std `sigma`. Optimal policy: action 0 at start, return 1.0.
* `slip`: with this probability the start action is replaced by the other one (transition noise).
* Many random paths (the m chains) merge into each bottleneck. This is where expected traces should
  help: one update at the bottleneck can move the values of all m chains, so the start-state values
  (which bootstrap on the chain values) become accurate sooner.

Observation: one-hot state id written into a 10x10xC grid (C = ceil(n_states / 100)), so the
MinAtar Q-network can be used unchanged. One-hot means no generalisation between chains, so any
cross-chain credit has to come from the trace, not from the network.

Use from the agents without code changes (this file must sit in the repo root):
    --env_type minatar --env_id multichain_env:MultiChain-m16-n8-s1.0-p0.0-v0
Registered ids: m in {1, 4, 16, 64}, n = 8, sigma in {0.0, 0.5, 1.0, 2.0}, slip in {0.0, 0.2}.

NOTE (tabular checks, 26 Oct): in this control version ET does NOT beat Q(lambda) once lambda is tuned.
Use the PREDICTION version below (MultiChainPred-*) for the deep test.

PREDICTION version (MultiChainPredEnv): the paper's original multi-chain. One action only, so the
Q-learning agents become TD(lambda) / ET(lambda) value predictors. start -> one of m chains at random
-> shared bottleneck -> terminal reward 1.0 + sigma * noise (all other rewards 0).
Use `true_values()` and `all_observations()` to measure value error over ALL states:
    --env_type minatar --env_id multichain_env:MultiChainPred-m32-n4-s1.0-v0
Registered ids: m in {1, 8, 32, 128}, n = 4, sigma in {0.0, 1.0}.
"""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

MEANS = (1.0, 0.6)  # mean terminal reward for group A (action 0) and group B (action 1)


class MultiChainEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, m=16, n=8, sigma=1.0, slip=0.0, num_actions=2):
        self.m, self.n, self.sigma, self.slip = m, n, sigma, slip
        # ids: 0 = start; group g chain c position i -> 1 + g*m*n + c*n + i; bottleneck g -> 1 + 2*m*n + g
        self.n_states = 1 + 2 * m * n + 2
        self.channels = math.ceil(self.n_states / 100)
        self.observation_space = spaces.Box(0.0, 1.0, (10, 10, self.channels), dtype=np.float32)
        self.action_space = spaces.Discrete(num_actions)
        self._rng = np.random.default_rng()
        self.state = 0

    def _obs(self):
        o = np.zeros((10, 10, self.channels), dtype=np.float32)
        k = self.state
        o[(k % 100) // 10, k % 10, k // 100] = 1.0
        return o

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state, self.group, self.chain, self.pos = 0, None, None, None
        return self._obs(), {}

    def step(self, action):
        reward, terminated = 0.0, False
        if self.state == 0:                                    # the only real decision
            g = int(action) % 2
            if self._rng.random() < self.slip:
                g = 1 - g
            self.group, self.chain, self.pos = g, int(self._rng.integers(self.m)), 0
            self.state = 1 + g * self.m * self.n + self.chain * self.n
        elif self.pos is not None and self.pos < self.n - 1:   # move along the chain
            self.pos += 1
            self.state += 1
        elif self.pos == self.n - 1:                           # chain end -> group bottleneck
            self.pos = None
            self.state = 1 + 2 * self.m * self.n + self.group
        else:                                                  # bottleneck -> terminal reward
            reward = MEANS[self.group] + self.sigma * float(self._rng.standard_normal())
            terminated = True
        return self._obs(), reward, terminated, False, {"group": self.group}


for _m in (1, 4, 16, 64):
    for _s in (0.0, 0.5, 1.0, 2.0):
        for _p in (0.0, 0.2):
            _id = f"MultiChain-m{_m}-n8-s{_s}-p{_p}-v0"
            if _id not in gym.envs.registry:
                gym.register(id=_id, entry_point=MultiChainEnv,
                             kwargs=dict(m=_m, n=8, sigma=_s, slip=_p), max_episode_steps=100)


class MultiChainPredEnv(gym.Env):
    """The paper's multi-chain as a single-action task (pure value prediction)."""
    metadata = {"render_modes": []}

    def __init__(self, m=32, n=4, sigma=1.0, mean=1.0, gamma=0.99):
        self.m, self.n, self.sigma, self.mean, self.gamma = m, n, sigma, mean, gamma
        # ids: 0 = start; chain c position i -> 1 + c*n + i; bottleneck -> 1 + m*n
        self.n_states = 1 + m * n + 1
        self.channels = math.ceil(self.n_states / 100)
        self.observation_space = spaces.Box(0.0, 1.0, (10, 10, self.channels), dtype=np.float32)
        self.action_space = spaces.Discrete(1)
        self._rng = np.random.default_rng()
        self.state = 0

    def _encode(self, k):
        o = np.zeros((10, 10, self.channels), dtype=np.float32)
        o[(k % 100) // 10, k % 10, k // 100] = 1.0
        return o

    def all_observations(self):
        """(n_states, 10, 10, C) array: the observation of every state, in state-id order."""
        return np.stack([self._encode(k) for k in range(self.n_states)])

    def true_values(self):
        """Exact discounted values of every state (same order as all_observations)."""
        v = np.zeros(self.n_states)
        bottleneck = self.n_states - 1
        v[bottleneck] = self.mean                       # its transition gives the terminal reward
        for i in range(self.n):                         # chain position i is (n - i) steps from bottleneck
            for c in range(self.m):
                v[1 + c * self.n + i] = self.gamma ** (self.n - i) * self.mean
        v[0] = self.gamma ** (self.n + 1) * self.mean
        return v

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.state, self.pos = 0, None
        return self._encode(self.state), {}

    def step(self, action):
        reward, terminated = 0.0, False
        bottleneck = self.n_states - 1
        if self.state == 0:
            self.pos = 0
            self.state = 1 + int(self._rng.integers(self.m)) * self.n
        elif self.state == bottleneck:
            reward = self.mean + self.sigma * float(self._rng.standard_normal())
            terminated = True
        elif self.pos < self.n - 1:
            self.pos += 1
            self.state += 1
        else:
            self.state = bottleneck
        return self._encode(self.state), reward, terminated, False, {}


for _m in (1, 8, 32, 128):
    for _s in (0.0, 1.0):
        _id = f"MultiChainPred-m{_m}-n4-s{_s}-v0"
        if _id not in gym.envs.registry:
            gym.register(id=_id, entry_point=MultiChainPredEnv,
                         kwargs=dict(m=_m, n=4, sigma=_s), max_episode_steps=100)


class MultiChainSparseEnv(MultiChainPredEnv):
    """Sparse version (as the open world in Fig. 1/2 of the ET paper): terminal reward +1 w.p. p, otherwise 0.
    Random numbers are drawn in the same order as make_paper_figures.simulate (chain, then reward), so a deep run and a
    tabular simulation with the same seed see exactly the same episodes and rewards."""

    def __init__(self, m=8, n=4, p=0.2, gamma=0.99):
        super().__init__(m=m, n=n, sigma=0.0, mean=p, gamma=gamma)
        self.p = p

    def step(self, action):
        if self.state == self.n_states - 1:                    # shared state -> terminal, sparse reward
            return self._encode(self.state), (1.0 if self._rng.random() < self.p else 0.0), True, False, {}
        return super().step(action)


for _m in (8, 32):
    for _p in (0.2, 0.5, 0.9):
        _id = f"MultiChainSparse-m{_m}-n4-p{_p}-v0"
        if _id not in gym.envs.registry:
            gym.register(id=_id, entry_point=MultiChainSparseEnv, kwargs=dict(m=_m, n=4, p=_p), max_episode_steps=100)
