"""Adds an --eval_values flag to algorithms/qrc.py and algorithms/qrc-et-algo3.py.

With the flag on (multi-chain PREDICTION env only), every logging step (1000 env steps) also logs:
  value_rmse         RMSE of the agent's value estimate over ALL states vs the exact values
  value_rmse_chains  same, chain states only
  value_start_err    signed error at the start state
in true (unscaled) reward units. Without the flag nothing changes.

How the estimate is computed: the agent sees normalized observations and scaled rewards, so every
state's observation is normalized with the agent's current observation statistics (without updating
them), the Q-network is applied, and the output is multiplied back by the current reward scale.

Run once from the repo root:  python apply_eval_patch.py
It refuses to change anything if the files don't look as expected, and is safe to run twice.
"""
import sys

FILES = ["algorithms/qrc.py", "algorithms/qrc-et-algo3.py"]

ARG_ANCHOR = "    use_spr: bool = False\n"
ARG_NEW = (
    "    eval_values: bool = False\n"
    '    """Multi-chain prediction env only: log value RMSE over ALL states vs the exact values every log step."""\n'
)

FUNC_ANCHOR = "def make_env(args, idx, run_name):\n"
FUNC_NEW = '''def value_error_metrics(env, agent_state):
    """Value error over all states of a MultiChainPred env, in true (unscaled) reward units."""
    base = env.unwrapped
    obs_stats = env.get_wrapper_attr("obs_stats")
    obs = (base.all_observations() - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8)
    ts = agent_state.train_state
    q = np.asarray(ts.apply_fn(ts.params, jnp.asarray(obs, dtype=jnp.float32)))[:, 0]
    reward_scale = float(np.sqrt(np.ravel(env.get_wrapper_attr("reward_stats").var)[0] + 1e-8))
    err = q * reward_scale - base.true_values()
    return {
        "value_rmse": float(np.sqrt(np.mean(err ** 2))),
        "value_rmse_chains": float(np.sqrt(np.mean(err[1:-1] ** 2))),
        "value_start_err": float(err[0]),
    }


'''

LOG_ANCHOR = '                "episodes": episodes,\n            }\n'
LOG_NEW = (
    "            if args.eval_values:\n"
    "                log_dict.update(value_error_metrics(env, agent_state))\n"
)

ok = True
for f in FILES:
    s = open(f).read()
    if "def value_error_metrics" in s:
        print(f"{f}: already patched, skipping")
        continue
    counts = {name: s.count(a) for name, a in [("args", ARG_ANCHOR), ("make_env", FUNC_ANCHOR), ("log_dict", LOG_ANCHOR)]}
    if any(c != 1 for c in counts.values()):
        print(f"{f}: NOT patched, anchors not found exactly once: {counts}")
        ok = False
        continue
    s = s.replace(ARG_ANCHOR, ARG_NEW + ARG_ANCHOR)
    s = s.replace(FUNC_ANCHOR, FUNC_NEW + FUNC_ANCHOR)
    s = s.replace(LOG_ANCHOR, LOG_ANCHOR + LOG_NEW)
    open(f, "w").write(s)
    print(f"{f}: patched")
sys.exit(0 if ok else 1)
