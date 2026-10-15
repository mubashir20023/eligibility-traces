"""Small actual-JAX regression checks, separate from the standard-library data audit."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "algorithms" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


et = load("published_et_agent", "qrc-et-algo3-lab.py")
qrc = load("published_qrc_baseline", "qrc.py")


def config(**overrides):
    args = et.Args(env_type="gridworld", grid_hidden=0, q_lr=.01,
                   et_paired_rng=True, et_zero_init=True, et_normalize_grad=True)
    return et.Config.from_dict({**asdict(args), **overrides})


def transition(t):
    return (jnp.eye(6)[t % 2], t % 2, jnp.eye(6)[(t + 1) % 2], float(t % 3 == 0))


class AgentChecks(unittest.TestCase):
    def close(self, a, b):
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
            np.testing.assert_allclose(x, y, atol=1e-6, rtol=1e-6)

    def test_eta_one_recovers_qrc_update(self):
        cfg = config(et_eta=1.0)
        a, _ = et.init_agent_state_qrc_agent(cfg, 2, (6,), jax.random.PRNGKey(19))
        b = qrc.AgentState(qrc.Config.from_dict(cfg.d), a.train_state, a.h_state,
                           a.h_trace, a.grad_h_trace, a.grad_q_trace)
        for t in range(20):
            flags = (t % 5 == 4, False, t % 4 == 3)
            a, _ = et.update_step_qrc_agent(a, transition(t), *flags)
            b, _ = qrc.update_step_qrc_agent(b, transition(t), *flags)
            self.close(a.train_state.params, b.train_state.params)
            self.close(a.h_state.params, b.h_state.params)

    def test_learned_prediction_changes_q_weights(self):
        a, _ = et.init_agent_state_qrc_agent(config(), 2, (6,), jax.random.PRNGKey(7))
        b, _ = et.init_agent_state_qrc_agent(config(et_zero_theta=True), 2, (6,), jax.random.PRNGKey(7))
        for t in range(20):
            a, _ = et.update_step_qrc_agent(a, transition(t), False, False, False)
            b, _ = et.update_step_qrc_agent(b, transition(t), False, False, False)
        delta = sum(float(jnp.sum(jnp.abs(x - y))) for x, y in zip(
            jax.tree.leaves(a.train_state.params), jax.tree.leaves(b.train_state.params), strict=True))
        self.assertGreater(delta, 1e-8)

    def test_episode_end_clears_traces(self):
        a, _ = et.init_agent_state_qrc_agent(config(), 2, (6,), jax.random.PRNGKey(7))
        a, _ = et.update_step_qrc_agent(a, transition(0), True, False, False)
        self.close(a.grad_q_trace, jax.tree.map(jnp.zeros_like, a.grad_q_trace))
        np.testing.assert_allclose(a.et_y_trace, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
