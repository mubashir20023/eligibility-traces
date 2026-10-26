"""Tests for the Algorithm-3 variant of QRC-ET (algorithms/qrc-et-algo3.py).

Run from repo root:  python -m pytest tests/test_et_algo3.py -q
Checks (per Mathieu's review of qrc-et.py):
  1. eta=1 -> ET switched off -> identical Q-params to the original qrc-et.py.
  2. Weight update at step t never depends on step t's Theta (only e_w_t).
  3. Theta's influence reaches the weights exactly one step later (via y_{t-1}).
  4. Theta is regressed on the WHOLE trace (every action column gets gradient);
     the original only supervised the taken column.
  5. Theta's loss target equals e_w_t = gamma*lambda*y_{t-1} + grad.
  6. Trace reset (terminal / non-greedy) zeroes y.
"""
import importlib.util, os, sys
import numpy as np
import jax, jax.numpy as jnp
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "algorithms", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ALGO3 = _load("qrc_et_algo3", "qrc-et-algo3.py")
ORIG = _load("qrc_et_orig", "qrc-et.py")
A, OBS = 4, (10, 10, 4)


def make(mod, **kw):
    args = mod.Args(env_type="minatar", env_id="MinAtar/Breakout-v1", total_timesteps=1000, **kw)
    cfg = mod.Config.from_args(args)
    st, _ = mod.init_agent_state_qrc_agent(cfg, A, OBS, jax.random.PRNGKey(0))
    return st


def trans(seed):
    r = np.random.RandomState(seed)
    return (jnp.array(r.rand(*OBS), jnp.float32), int(r.randint(A)),
            jnp.array(r.rand(*OBS), jnp.float32), float(r.randn()))


def step(mod, st, seed, term=False):
    return mod.update_step_qrc_agent(st, trans(seed), term, False, False)


def leaves(t):
    return jax.tree_util.tree_leaves(t)


def close(a, b, tol=1e-6):
    return all(np.allclose(x, y, atol=tol) for x, y in zip(leaves(a), leaves(b)))


def perturb_theta(st, scale=1.0):
    p = jax.tree_util.tree_map(lambda x: x + scale * jax.random.normal(jax.random.PRNGKey(7), x.shape),
                               st.et_state.params)
    return st._replace(et_state=st.et_state.replace(params=p))


def test_eta1_matches_original():
    s3, so = make(ALGO3, et_eta=1.0), make(ORIG, et_eta=1.0)
    for i in range(5):
        s3, _ = step(ALGO3, s3, i)
        so, _ = step(ORIG, so, i)
    assert close(s3.train_state.params, so.train_state.params)


def test_weight_update_ignores_current_theta():
    base = make(ALGO3, et_eta=0.3)
    base, _ = step(ALGO3, base, 0)          # build some trace history
    a, _ = step(ALGO3, base, 1)
    b, _ = step(ALGO3, perturb_theta(base), 1)
    assert close(a.train_state.params, b.train_state.params), "step-t weights depend on step-t Theta"
    assert not np.allclose(a.et_y_trace, b.et_y_trace), "y_t should carry Theta"


def test_original_does_depend_on_current_theta():
    base = make(ORIG, et_eta=0.3)
    base, _ = step(ORIG, base, 0)
    a, _ = step(ORIG, base, 1)
    b, _ = step(ORIG, perturb_theta(base), 1)
    assert not close(a.train_state.params, b.train_state.params)


def test_theta_reaches_weights_one_step_later():
    base = make(ALGO3, et_eta=0.3)
    base, _ = step(ALGO3, base, 0)
    a, _ = step(ALGO3, base, 1)
    b, _ = step(ALGO3, perturb_theta(base), 1)
    a2, _ = step(ALGO3, a, 2)
    b2, _ = step(ALGO3, b, 2)
    assert not close(a2.train_state.params, b2.train_state.params)


def theta_grad_rows(mod, st, seed):
    """Gradient norm per action row of Theta's output layer for one update."""
    new, _ = step(mod, st, seed)
    k0 = st.et_state.params["params"]["Theta"]["kernel"]
    k1 = new.et_state.params["params"]["Theta"]["kernel"]
    d = np.array(k1 - k0).reshape(k0.shape[0], A, -1)  # (in, action, feat)
    return np.linalg.norm(d, axis=(0, 2))


def test_theta_supervised_on_all_columns():
    st3 = make(ALGO3, et_eta=0.3)
    # give y_{t-1} nonzero so untaken columns have non-zero targets
    st3, _ = step(ALGO3, st3, 0)
    rows3 = theta_grad_rows(ALGO3, st3, 1)
    assert (rows3 > 0).all(), rows3
    so = make(ORIG, et_eta=0.3)
    so, _ = step(ORIG, so, 0)
    rowso = theta_grad_rows(ORIG, so, 1)
    assert (rowso > 0).sum() == 1, rowso   # only the taken column


def test_theta_target_is_e_w_t():
    st = make(ALGO3, et_eta=0.3)
    st, _ = step(ALGO3, st, 0)
    obs, action, nobs, r = trans(1)
    cfg = st.agent_config
    head = ALGO3.get_head_name(cfg)
    qg = jax.grad(lambda p: st.train_state.apply_fn(p, obs)[action])(st.train_state.params)
    hk = qg["params"][head]["kernel"]
    e_w = cfg.gamma * cfg.lamda * st.et_y_trace + hk
    z = st.et_state.apply_fn(st.et_state.params, hk[:, action])
    expect = float(jnp.sum(jnp.mean((z - e_w.T) ** 2, axis=-1)))
    _, m = step(ALGO3, st, 1)
    assert np.isclose(float(m["et_trace_loss"]), expect, rtol=1e-4)


def test_reset_zeroes_y():
    st = make(ALGO3, et_eta=0.3)
    st, _ = step(ALGO3, st, 0)
    assert np.abs(st.et_y_trace).sum() > 0
    st, _ = step(ALGO3, st, 1, term=True)
    assert np.abs(st.et_y_trace).sum() == 0


def test_zero_theta_ablation_ignores_theta_and_scales_trace():
    a = make(ALGO3, et_eta=0.3, et_zero_theta=True)
    b = perturb_theta(a, scale=3.0)
    for i in range(4):
        a, _ = step(ALGO3, a, i)
        b, _ = step(ALGO3, b, i)
    assert close(a.train_state.params, b.train_state.params), "zero-Theta ablation still depends on Theta"
    # y_t = eta * e_w_t, e_w_t = gamma*lambda*y_{t-1} + grad  =>  decay of real gradient is eta*gamma*lambda
    c = make(ALGO3, et_eta=0.3, et_zero_theta=True)
    c, _ = step(ALGO3, c, 0)
    y1 = np.array(c.et_y_trace)
    c, _ = step(ALGO3, c, 1)
    assert np.abs(y1).sum() > 0


def test_raw_target_option_uses_theta_independent_trace():
    st = make(ALGO3, et_eta=0.3, et_algo3_target="raw")
    st, _ = step(ALGO3, st, 0)
    obs, action, nobs, r = trans(1)
    cfg = st.agent_config; head = ALGO3.get_head_name(cfg)
    qg = jax.grad(lambda p: st.train_state.apply_fn(p, obs)[action])(st.train_state.params)
    hk = qg["params"][head]["kernel"]
    raw = cfg.gamma * cfg.lamda * st.grad_q_trace["params"][head]["kernel"] + hk
    z = st.et_state.apply_fn(st.et_state.params, hk[:, action])
    expect = float(jnp.sum(jnp.mean((z - raw.T) ** 2, axis=-1)))
    _, m = step(ALGO3, st, 1)
    assert np.isclose(float(m["et_trace_loss"]), expect, rtol=1e-4)


def test_no_lag_option_uses_current_theta():
    base = make(ALGO3, et_eta=0.3, et_algo3_lag=False)
    base, _ = step(ALGO3, base, 0)
    a, _ = step(ALGO3, base, 1)
    b, _ = step(ALGO3, perturb_theta(base), 1)
    assert not close(a.train_state.params, b.train_state.params)


def test_paired_rng_reproduces_plain_qrc_trajectory():
    """With et_eta=1 and et_paired_rng=True the ET agent must follow plain qrc.py action-for-action."""
    QRC = _load("qrc_plain", "qrc.py")
    def run(mod, steps=300, seed=1, **kw):
        args = mod.Args(env_type="minatar", env_id="MinAtar/Freeway-v1", seed=seed, total_timesteps=5_000_000, **kw)
        cfg = mod.Config.from_args(args); env = mod.make_env(args, 0, "t")
        obs, _ = env.reset(seed=seed); obs = np.array(obs); A = int(env.action_space.n)
        rng = jax.random.PRNGKey(seed)
        st, rng = mod.QRCAgent.init_state(cfg, A, obs.shape, rng)
        sched = mod.get_linear_epsilon_schedule(args); acts = []
        for t in range(steps):
            a, ng, rng = mod.QRCAgent.step(st, obs, A, sched(t), rng); a = int(a); ng = bool(ng)
            nobs, r, term, trunc, _ = env.step(a); nobs = np.array(nobs)
            st, _ = mod.QRCAgent.update(st, (obs, a, nobs.copy(), r), term, trunc, ng)
            if term or trunc:
                nobs, _ = env.reset(); nobs = np.array(nobs)
            acts.append(a); obs = nobs
        return np.array(acts), st.train_state.params
    aq, pq = run(QRC)
    ap_, pp = run(ALGO3, et_eta=1.0, et_paired_rng=True)
    assert np.array_equal(aq, ap_)
    assert close(pq, pp, tol=1e-5)
    an, _ = run(ALGO3, et_eta=1.0, et_paired_rng=False)
    assert not np.array_equal(aq, an), "old init should NOT be paired with qrc.py"


def test_stochasticity_knobs_default_off_and_wired():
    for mod in (ALGO3, _load("qrc_plain2", "qrc.py")):
        a0 = mod.Args(env_type="minatar", env_id="MinAtar/Freeway-v1")
        assert a0.sticky_action_prob is None and a0.reward_noise_std == 0.0
        e = mod.make_env(mod.Args(env_type="minatar", env_id="MinAtar/Freeway-v1", sticky_action_prob=0.5), 0, "t")
        assert e.get_wrapper_attr("game").sticky_action_prob == 0.5
        e = mod.make_env(mod.Args(env_type="minatar", env_id="MinAtar/Asterix-v1"), 0, "t")
        assert e.get_wrapper_attr("game").sticky_action_prob == 0.1       # MinAtar default untouched


def test_reward_noise_reaches_agent_but_not_logged_returns():
    mod = ALGO3
    e = mod.make_env(mod.Args(env_type="minatar", env_id="MinAtar/Freeway-v1", reward_noise_std=1.0, seed=3), 0, "t")
    e.reset(seed=3)
    rs = []
    for _ in range(200):
        _, r, term, trunc, _ = e.step(0)
        rs.append(r)
        if term or trunc:
            e.reset()
    assert np.std(rs) > 0.1                                  # agent sees noisy rewards
    rq = list(e.get_wrapper_attr("return_queue"))
    assert all(float(x) == round(float(x)) for x in rq)      # logged episode returns are true (integer) returns


def test_gridworld_network_runs_and_theta_matches_state_space():
    args = ALGO3.Args(env_type="gridworld", env_id="grid", total_timesteps=1000, q_lr=0.05, opt="sgd", et_eta=0.3,
                      et_algo3=True, et_paired_rng=True, grid_hidden=0)
    cfg = ALGO3.Config.from_args(args)
    st, rng = ALGO3.QRCAgent.init_state(cfg, 4, (25,), jax.random.PRNGKey(0))
    assert st.et_y_trace.shape == (25, 4)               # one-hot features, 4 actions
    x = np.zeros(25, np.float32); x[3] = 1.0; x2 = np.zeros(25, np.float32); x2[4] = 1.0
    st2, _ = ALGO3.QRCAgent.update(st, (x, 1, x2, 1.0), True, False, False)
    q = st2.train_state.apply_fn(st2.train_state.params, x)
    assert float(q[1]) > 0.0                            # reward 1 at termination raises q(s=3, a=1)
    assert float(q[0]) == 0.0                           # untouched action unchanged
