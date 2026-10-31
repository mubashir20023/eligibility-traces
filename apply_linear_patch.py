"""Adds a no-convolution option to algorithms/qrc.py and algorithms/qrc-et-algo3.py (Peter's suggestion).

  --net_arch linear     Q(s) = W x(s): one linear layer on the flattened observation, NO bias, zero initialisation.
                        With the multi-chain environments x(s) is a one-hot state indicator, so nothing is shared between
                        states (no conv filters, hidden layers or bias) -- the deep agent becomes tabular QRC / QRC-ET.
                        For QRC-ET also pass --et_head_name Dense_0 (the only layer is the head).
  --no-obs_norm         skip the running mean/std observation normalisation (it adds a small shared offset to every
                        input, which is a weak form of generalisation). Default keeps it on, so nothing else changes.

Run from the repo root AFTER apply_eval_patch.py and apply_snapshot_patch.py:  python apply_linear_patch.py
Safe to run twice.
"""
import sys

FILES = ["algorithms/qrc.py", "algorithms/qrc-et-algo3.py"]

ARG_ANCHOR = "    eval_values: bool = False\n"
ARG_NEW = ('    obs_norm: bool = True\n'
           '    """MinAtar/multi-chain: normalise observations with a running mean/std (False = raw one-hot input)."""\n')

CLASS_ANCHOR = "def make_env(args, idx, run_name):\n"
CLASS_NEW = '''import flax.linen as nn  # needed for the class below (qrc.py does not import it)


class OneHotLinearQNetwork(nn.Module):
    """No convolution, no hidden layer, no bias: Q(s) = W x(s) on the flattened observation (tabular for one-hot x)."""
    action_dim: int

    @nn.compact
    def __call__(self, x):
        no_batch = x.ndim == 3
        if no_batch:
            x = x[None]
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.action_dim, use_bias=False, kernel_init=nn.initializers.zeros, name="Dense_0")(x)
        return x[0] if no_batch else x


def _normalise_obs(env, x):
    """Apply the agent's observation normalisation if it is on (read-only)."""
    try:
        st = env.get_wrapper_attr("obs_stats")
    except AttributeError:
        return x
    return (x - st.mean) / np.sqrt(st.var + 1e-8)


'''

NET_OLD = ('        elif agent_config.env_type == "minatar":\n'
           '            network = MinAtarQNetwork(**net_kwargs)\n')
NET_NEW = ('        elif agent_config.env_type == "minatar" and agent_config.net_arch == "linear":\n'
           '            network = OneHotLinearQNetwork(action_dim=action_dim)\n'
           '            init_x = jnp.zeros(obs_shape)\n'
           '        elif agent_config.env_type == "minatar":\n'
           '            network = MinAtarQNetwork(**net_kwargs)\n')

NORM_OLD = "        env = normalization.NormalizeObservation(env)\n\n        return env\n"
NORM_NEW = "        if args.obs_norm:\n            env = normalization.NormalizeObservation(env)\n\n        return env\n"

EVAL_OLD = ('    obs_stats = env.get_wrapper_attr("obs_stats")\n'
            '    obs = (base.all_observations() - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8)\n')
EVAL_NEW = '    obs = _normalise_obs(env, base.all_observations())\n'
SNAP_OLD = ('    obs_stats = env.get_wrapper_attr("obs_stats")\n'
            '    obs = jnp.asarray((base.all_observations() - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8), dtype=jnp.float32)\n')
SNAP_NEW = '    obs = jnp.asarray(_normalise_obs(env, base.all_observations()), dtype=jnp.float32)\n'

ok = True
for f in FILES:
    s = open(f).read()
    if "class OneHotLinearQNetwork" in s:
        if "import flax.linen as nn  # needed" not in s:                     # repair an earlier version of this patch
            s = s.replace("class OneHotLinearQNetwork(nn.Module):",
                          "import flax.linen as nn  # needed for the class below (qrc.py does not import it)\n\n\n"
                          "class OneHotLinearQNetwork(nn.Module):")
            open(f, "w").write(s)
            print(f"{f}: repaired (added missing import)"); continue
        print(f"{f}: already patched, skipping"); continue
    need = {"args": ARG_ANCHOR, "make_env": CLASS_ANCHOR, "network": NET_OLD, "obs_norm": NORM_OLD,
            "eval patch": EVAL_OLD, "snapshot patch": SNAP_OLD}
    counts = {k: s.count(v) for k, v in need.items()}
    if any(c != 1 for c in counts.values()):
        print(f"{f}: NOT patched (apply_eval_patch.py and apply_snapshot_patch.py first?) anchors: {counts}"); ok = False; continue
    s = (s.replace(ARG_ANCHOR, ARG_NEW + ARG_ANCHOR).replace(CLASS_ANCHOR, CLASS_NEW + CLASS_ANCHOR)
          .replace(NET_OLD, NET_NEW).replace(NORM_OLD, NORM_NEW).replace(EVAL_OLD, EVAL_NEW).replace(SNAP_OLD, SNAP_NEW))
    open(f, "w").write(s)
    print(f"{f}: patched")
sys.exit(0 if ok else 1)
