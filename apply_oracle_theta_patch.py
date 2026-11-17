"""Adds two options for R40c (Peter's question: is theta's representation the bottleneck?).

  --q_channels N              linear network (--net_arch linear): the Q and h networks use only the first N
                              observation channels (0 = all, the default).
  --et_theta_input extra_channels
                              qrc-et-algo3.py: theta reads the observation channels AFTER the first q_channels
                              (e.g. a one-hot state id) instead of the Q-head features. theta still predicts the
                              expected trace of the Q-head features, so only theta's input changes.
With multichain_env MultiChainCodeId-* (shared code in the first channels, one-hot state id after it):
  --net_arch linear --q_channels 1 --et_theta_input extra_channels   ->  Q on the shared code, theta on the state id.

Run from the repo root AFTER apply_qlambda_patch.py:  python apply_oracle_theta_patch.py   (safe to run twice)
"""
import sys

FILES = ["algorithms/qrc.py", "algorithms/qrc-et-algo3.py"]
ARG_ANCHOR = "    q_lambda: bool = False\n"
ARG_NEW = ('    q_channels: int = 0\n'
           '    """Linear network: Q and h use only the first N observation channels (0 = all)."""\n'
           '    et_theta_input: str = "features"\n'
           '    """QRC-ET: theta input -- "features" (Q-head features) or "extra_channels" (channels after q_channels)."""\n')
CLS_OLD = ('    action_dim: int\n\n    @nn.compact\n    def __call__(self, x):\n        no_batch = x.ndim == 3\n'
           '        if no_batch:\n            x = x[None]\n        x = x.reshape((x.shape[0], -1))\n'
           '        x = nn.Dense(self.action_dim, use_bias=False, kernel_init=nn.initializers.zeros, name="Dense_0")(x)\n')
CLS_NEW = ('    action_dim: int\n    q_channels: int = 0\n\n    @nn.compact\n    def __call__(self, x):\n        no_batch = x.ndim == 3\n'
           '        if no_batch:\n            x = x[None]\n        if self.q_channels:\n            x = x[..., :self.q_channels]\n'
           '        x = x.reshape((x.shape[0], -1))\n'
           '        x = nn.Dense(self.action_dim, use_bias=False, kernel_init=nn.initializers.zeros, name="Dense_0")(x)\n')
NET_OLD = "            network = OneHotLinearQNetwork(action_dim=action_dim)\n"
NET_NEW = "            network = OneHotLinearQNetwork(action_dim=action_dim, q_channels=agent_config.q_channels)\n"
# qrc-et-algo3.py only
INIT_OLD = "    et_params = et_module.init(_rng, jnp.zeros(feat_dim))\n"
INIT_NEW = ('    theta_in = feat_dim\n'
            '    if agent_config.et_theta_input == "extra_channels":  # R40c: theta reads the extra channels\n'
            '        theta_in = int(np.prod(obs_shape[:-1]) * (obs_shape[-1] - agent_config.q_channels))\n'
            '        print(f"theta input: observation channels {agent_config.q_channels}+ ({theta_in} values)")\n'
            '    et_params = et_module.init(_rng, jnp.zeros(theta_in))\n')
XS_OLD = '    x_s = q_grads["params"][head_name]["kernel"][:, action]  # = x(s)\n'
XS_NEW = XS_OLD + ('    if config.et_theta_input == "extra_channels":  # R40c: theta reads the extra (e.g. state-id) channels\n'
                   '        x_s = jax.lax.stop_gradient(obs[..., config.q_channels:]).reshape(-1)\n')

ok = True
for f in FILES:
    s = open(f).read()
    if "q_channels: int" in s:
        print(f"{f}: already patched, skipping"); continue
    reps = [(ARG_ANCHOR, ARG_NEW + ARG_ANCHOR), (CLS_OLD, CLS_NEW), (NET_OLD, NET_NEW)]
    if f.endswith("qrc-et-algo3.py"):
        reps += [(INIT_OLD, INIT_NEW), (XS_OLD, XS_NEW)]
    counts = [s.count(a) for a, _ in reps]
    if any(c != 1 for c in counts):
        print(f"{f}: NOT patched (run apply_qlambda_patch.py first?) anchors {counts}"); ok = False; continue
    for a, b in reps:
        s = s.replace(a, b)
    open(f, "w").write(s)
    print(f"{f}: patched (--q_channels, --et_theta_input)")
sys.exit(0 if ok else 1)
