"""Adds three options to algorithms/qrc.py and algorithms/qrc-et-algo3.py for Peter's two MinAtar follow-ups.

  --net_arch mlp        (MinAtar) no convolution: flatten -> Dense(--mlp_hidden, default 128) -> activation -> Q head.
                        Same activation, LayerNorm and sparse init as the conv network; the head is "Dense_1" as there.
                        (--net_arch linear already exists: Q = W x on the flattened observation; ET needs --et_head_name Dense_0.)
  --init_from PATH      start from the representation of a trained agent: copies every layer EXCEPT the Q head
                        (last Dense) from PATH (a final "<exp_name>.cleanrl_model" file, which stores [q params, h params]),
                        for both the Q network and QRC's h network. Heads stay freshly initialised, optimiser state, traces
                        and Theta start from zero. Ignored when --resume finds this run's own checkpoint.
  --freeze_body         keep every layer except the head fixed (Q and h networks): their updates are set to zero.

Run from the repo root AFTER the earlier patches (eval, snapshot, linear, split):  python apply_r38_r39_patch.py
Safe to run twice; nothing changes unless the new flags are used.
"""
import sys

FILES = ["algorithms/qrc.py", "algorithms/qrc-et-algo3.py"]
MARK = "def load_body("

ARG_ANCHOR = '    net_arch: str = "minatar"\n'
ARG_NEW = ('    mlp_hidden: int = 128\n'
           '    """MinAtar --net_arch mlp: width of the single hidden layer."""\n'
           '    init_from: str = ""\n'
           '    """Path to a trained agent\'s .cleanrl_model: copy all layers except the Q head (Q and h networks)."""\n'
           '    freeze_body: bool = False\n'
           '    """Keep all layers except the head fixed (Q and h networks)."""\n')

CLASS_ANCHOR = "def make_env(args, idx, run_name):\n"
CLASS_NEW = '''from networks.layer_norm import layer_norm as _layer_norm  # used by MinAtarMLPQNetwork
from networks.ActorCritic import act_funcs as _act_funcs


class MinAtarMLPQNetwork(nn.Module):
    """MinAtar Q network without convolution: flatten -> Dense(hidden) -> activation -> Dense(action_dim)."""
    action_dim: int
    hidden: int
    layer_norm: bool
    activation: str
    kernel_init: Callable

    @nn.compact
    def __call__(self, x):
        no_batch = x.ndim == 3
        if no_batch:
            x = x[None]
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden, kernel_init=self.kernel_init)(x)
        if self.layer_norm:
            x = _layer_norm(x)
        x = _act_funcs[self.activation](x)
        x = nn.Dense(self.action_dim, kernel_init=self.kernel_init, bias_init=nn.initializers.zeros)(x)
        return x[0] if no_batch else x


def _head_name(params):
    keys = [k for k in params["params"] if k.startswith("Dense")]
    return max(keys, key=lambda k: int(k.split("_")[-1]))


def mask_body(update):
    """Zero the update of every layer except the head (used by --freeze_body)."""
    head = _head_name(update)
    return jax.tree_util.tree_map_with_path(
        lambda path, x: x if any(getattr(p, "key", None) == head for p in path) else jnp.zeros_like(x), update)


def load_body(agent_state, path):
    """--init_from: copy all layers except the head from a saved [q params, h params] model file."""
    with open(path, "rb") as f:
        data = f.read()
    tq, th = agent_state.train_state, agent_state.h_state
    src_q, src_h = flax.serialization.from_bytes([tq.params, th.params], data)

    def merge(new, src):
        head = _head_name(new)
        out, s = flax.core.unfreeze(new), flax.core.unfreeze(src)
        copied = []
        for k in out["params"]:
            if k != head:
                out["params"][k] = s["params"][k]
                copied.append(k)
        print(f"  init_from: copied {copied}, kept fresh head {head}")
        return flax.core.freeze(out) if isinstance(new, flax.core.FrozenDict) else out

    q, h = merge(tq.params, src_q), merge(th.params, src_h)
    tq = tq.replace(params=q, opt_state=tq.tx.init(q))
    th = th.replace(params=h, opt_state=th.tx.init(h))
    print(f"Loaded representation from {path}")
    return agent_state._replace(train_state=tq, h_state=th)


'''

NET_ANCHOR = '        elif agent_config.env_type == "minatar" and agent_config.net_arch == "linear":\n'
NET_NEW = ('        elif agent_config.env_type == "minatar" and agent_config.net_arch == "mlp":\n'
           '            network = MinAtarMLPQNetwork(action_dim=action_dim, hidden=agent_config.mlp_hidden,\n'
           '                                         layer_norm=agent_config.layer_norm, activation=agent_config.activation,\n'
           '                                         kernel_init=net_kwargs["kernel_init"])\n'
           '            init_x = jnp.zeros(obs_shape)\n')

INIT_ANCHOR = "    agent_state, rng = agent.init_state(agent_config, action_dim, obs.shape, rng)\n"
INIT_NEW = ("    if args.init_from:\n"
            "        agent_state = load_body(agent_state, args.init_from)\n")

Q_ANCHOR = "    q_train_state = train_state.apply_gradients(\n"
Q_NEW = ("    if config.freeze_body:\n"
         "        q_update = mask_body(q_update)\n")
H_ANCHOR = "    h_train_state = agent_state.h_state.apply_gradients(grads=h_update)\n"
H_NEW = ("    if config.freeze_body:\n"
         "        h_update = mask_body(h_update)\n")

ok = True
for f in FILES:
    s = open(f).read()
    if MARK in s:
        print(f"{f}: already patched, skipping"); continue
    need = {"args": ARG_ANCHOR, "make_env": CLASS_ANCHOR, "network": NET_ANCHOR, "init": INIT_ANCHOR,
            "q update": Q_ANCHOR, "h update": H_ANCHOR}
    counts = {k: s.count(v) for k, v in need.items()}
    if any(c != 1 for c in counts.values()) or "import flax.linen as nn" not in s:
        print(f"{f}: NOT patched (run the earlier patches first?) anchors: {counts}"); ok = False; continue
    s = (s.replace(ARG_ANCHOR, ARG_ANCHOR + ARG_NEW).replace(CLASS_ANCHOR, CLASS_NEW + CLASS_ANCHOR)
          .replace(NET_ANCHOR, NET_NEW + NET_ANCHOR).replace(INIT_ANCHOR, INIT_ANCHOR + INIT_NEW)
          .replace(Q_ANCHOR, Q_NEW + Q_ANCHOR).replace(H_ANCHOR, H_NEW + H_ANCHOR))
    open(f, "w").write(s)
    print(f"{f}: patched (--net_arch mlp, --init_from, --freeze_body)")
sys.exit(0 if ok else 1)
