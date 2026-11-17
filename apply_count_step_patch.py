"""R42 patch: --et_count_step for algorithms/qrc-et-algo3.py   (run once from stream-rep-rl; safe to run again).

Adds a visit-count step size for theta, as in the tabular experiments of the expected-traces paper:
    --et_count_step d   theta's step for every input feature = 1 / (visits of that feature) ** d
                        d = 1 makes theta the running mean of the traces seen in each state (one-hot input).
                        0 (default) keeps the constant step --et_lr, and nothing changes for any existing run.
Why: with a constant step of 0.1, theta needs about 20 visits of a state before its expected trace is learned.
With many chains (m = 128) a chain state is visited only 8 times in the first 1,024 episodes, so QRC-ET starts slowly.
"""
import sys

path = "algorithms/qrc-et-algo3.py"
src = open(path).read()
if "et_count_step" in src:
    print("already patched (--et_count_step)")
    sys.exit(0)

edits = [
    # 1) the option
    ('''    et_split: bool = False
    """R35. Theta predicts only the past part of the trace; the head update adds the exact current gradient."""
''',
     '''    et_split: bool = False
    """R35. Theta predicts only the past part of the trace; the head update adds the exact current gradient."""
    et_count_step: float = 0.0
    """R42. If > 0: theta's step for every input feature is 1 / (visits of that feature) ** et_count_step, as in the
    tabular experiments of the expected-traces paper (1.0 = running mean of the traces). 0 = constant step et_lr."""
'''),
    # 2) theta keeps a visit count of its input features
    ('''    feat_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, x_s):
        no_batch = x_s.ndim == 1
        if no_batch:
            x_s = x_s[None]
''',
     '''    feat_dim: int
    action_dim: int
    count_step: bool = False

    @nn.compact
    def __call__(self, x_s):
        no_batch = x_s.ndim == 1
        if no_batch:
            x_s = x_s[None]
        if self.count_step:  # R42: visits of every input feature (updated by the agent, not by the loss)
            self.param("count", nn.initializers.zeros, (x_s.shape[-1],))
'''),
    ('''    et_module = ETProjection(feat_dim=feat_dim, action_dim=action_dim)
''',
     '''    et_module = ETProjection(feat_dim=feat_dim, action_dim=action_dim, count_step=agent_config.et_count_step > 0)
'''),
    # 3) the update: rescale theta's gradient so that every visit moves z a fraction 1 / visits ** d towards its target
    ('''    )(et_state.params)
    new_et_state = et_state.apply_gradients(grads=et_grads)
''',
     '''    )(et_state.params)
    if config.et_count_step > 0:  # R42: theta step = 1 / (visits of each input feature) ** et_count_step
        seen = jnp.abs(jax.lax.stop_gradient(x_s))
        visits = et_state.params["params"]["count"] + seen
        row = (head_grad_t.shape[0] / (2.0 * jnp.maximum(jnp.sum(seen ** 2), 1e-8))
               * jnp.maximum(visits, 1.0) ** (-config.et_count_step) / config.et_lr)
        et_grads = {"params": {"Theta": {"kernel": et_grads["params"]["Theta"]["kernel"] * row[:, None]},
                               "count": -seen / config.et_lr}}
    new_et_state = et_state.apply_gradients(grads=et_grads)
'''),
]
for old, new in edits:
    if src.count(old) != 1:
        sys.exit(f"PATCH FAILED: expected exactly one match for:\n{old}\nNothing was changed.")
    src = src.replace(old, new)
open(path, "w").write(src)
print("patched (--et_count_step)")
