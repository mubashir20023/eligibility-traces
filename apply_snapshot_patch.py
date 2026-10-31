"""Adds --snapshot_every N and --snapshot_episodes "1,3,10,..." to algorithms/qrc.py and algorithms/qrc-et-algo3.py
(needs --eval_values). Upgrades an earlier version of this patch automatically.
Also saves diagnostics (diag_*): trace diversity, feature cosine / effective rank, kernel spread, head share, z accuracy.

Every N steps it appends to <log_dir>/value_snapshots.npz, and right after each listed episode ends it appends to
<log_dir>/value_snapshots_episodes.npz, for the MultiChainPred environment:
  step, episode        environment step and episode count of the snapshot
  values   (k, S)      the agent's value estimate of every state (true reward units)
  true     (S,)        exact values
  credit_trace (k, S)  how much ONE head update at the bottleneck would change each state's value when the
                       update uses the ordinary lambda-trace of the path start -> chain 0 -> bottleneck
  credit_agent (k, S)  the same, but with the trace THIS agent actually uses at that step
                       (QRC: ordinary trace; QRC-ET: Algorithm-3 recursion with Theta, honouring et_eta and et_algo3_lag)
  credit_zb    (k, S)  ET only: credit of the bottleneck's own expected trace z(bottleneck)
Credit maps are x(s) . u for update direction u (head layer), i.e. the change of v(s) per unit TD error and step size.
Snapshots are pure reads: they do not change training or the random-number streams.

Run once from the repo root:  python apply_snapshot_patch.py   (safe to run twice)
"""
import sys

FILES = ["algorithms/qrc.py", "algorithms/qrc-et-algo3.py"]
ARG_ANCHOR = '    """Multi-chain prediction env only: log value RMSE over ALL states vs the exact values every log step."""\n'
ARG_V1 = ("    snapshot_every: int = 0\n"
          '    """Multi-chain prediction env only (with eval_values): save per-state values and credit maps every N steps (0 = off)."""\n')
ARG_NEW = ARG_V1 + ("    snapshot_episodes: str = \"\"\n"
                    '    """Multi-chain prediction env only (with eval_values): comma-separated episode numbers to snapshot, e.g. 1,3,10,100."""\n')
FUNC_ANCHOR = "def make_env(args, idx, run_name):\n"
FUNC_NEW = '''_SNAPSHOTS = {}


def save_value_snapshot(env, agent_state, t, args, episode=0):
    """Per-state values and one-update credit maps for MultiChainPred (Fig-3-style plots). Read-only."""
    base = env.unwrapped
    obs_stats = env.get_wrapper_attr("obs_stats")
    obs = jnp.asarray((base.all_observations() - obs_stats.mean) / np.sqrt(obs_stats.var + 1e-8), dtype=jnp.float32)
    ts = agent_state.train_state
    scale = float(np.sqrt(np.ravel(env.get_wrapper_attr("reward_stats").var)[0] + 1e-8))
    values = np.asarray(ts.apply_fn(ts.params, obs))[:, 0] * scale
    inner = ts.params["params"]
    head = sorted([k for k in inner if k.startswith("Dense")], key=lambda k: int(k.split("_")[-1]))[-1]

    def q0(kernel, o):
        p = dict(ts.params)
        p["params"] = {**inner, head: {**inner[head], "kernel": kernel}}
        return ts.apply_fn(p, o)[0]

    X = np.asarray(jax.vmap(jax.grad(q0), in_axes=(None, 0))(inner[head]["kernel"], obs))[:, :, 0]   # (S, feat)
    S, n = base.n_states, base.n
    path = [0] + [1 + i for i in range(n)] + [S - 1]          # start -> chain 0 -> bottleneck
    gl = args.gamma * args.lamda
    e = np.zeros(X.shape[1])
    for s in path:
        e = gl * e + X[s]
    credit_trace = X @ e
    credit_agent, credit_zb = credit_trace, np.full(S, np.nan)
    et = getattr(agent_state, "et_state", None)
    if et is not None and getattr(args, "et_algo3", False):
        z = lambda s: np.asarray(et.apply_fn(et.params, jnp.asarray(X[s])))[0]
        theta_off = getattr(args, "et_zero_theta", False)
        y = np.zeros(X.shape[1])
        for s in path:                                        # Algorithm 3: e_w = gl*y + x;  y = (1-eta) z + eta e_w
            e_w = gl * y + X[s]
            zs = np.zeros_like(e_w) if theta_off else z(s)
            y = (1 - args.et_eta) * zs + args.et_eta * e_w
        u = e_w if getattr(args, "et_algo3_lag", True) else y
        credit_agent, credit_zb = X @ u, X @ z(S - 1)
    # ---- diagnostics: why can (or can't) an expected trace differ from the sampled trace here?
    m_ = base.m
    def path_trace(c, upto):                                     # ordinary trace along chain c, ending at state `upto`
        pth = [0] + [1 + c * n + i for i in range(n)] + [S - 1]
        e_ = np.zeros(X.shape[1])
        for s_ in pth[:pth.index(upto) + 1]:
            e_ = gl * e_ + X[s_]
        return e_
    traces = np.array([path_trace(c, S - 1) for c in range(m_)])   # trace at the shared state, one per chain
    oracle = traces.mean(0)                                        # true expected trace E[e | shared state]
    diag = {"trace_diversity": float(np.mean(np.sum((traces - oracle) ** 2, 1)) / (np.sum(oracle ** 2) + 1e-12))}
    Xc = X[1:S - 1] / (np.linalg.norm(X[1:S - 1], axis=1, keepdims=True) + 1e-12)
    cos = Xc @ Xc.T
    diag["feature_cosine"] = float(cos[~np.eye(len(cos), dtype=bool)].mean())
    sv = np.linalg.svd(X, compute_uv=False); pr = sv / sv.sum()
    diag["feature_eff_rank"] = float(np.exp(-np.sum(pr * np.log(pr + 1e-12))))
    g_b = jax.grad(lambda p: ts.apply_fn(p, obs[S - 1])[0])(ts.params)          # full gradient at the shared state
    _, k_row = jax.jvp(lambda p: ts.apply_fn(p, obs)[:, 0], (ts.params,), (g_b,))
    k_row = np.asarray(k_row)                                                     # NTK row K(s, shared)
    diag["kernel_spread"] = float(np.mean(k_row[1:S - 1]) / (k_row[S - 1] + 1e-12))
    diag["head_share"] = float((X[S - 1] @ X[S - 1]) / (k_row[S - 1] + 1e-12))
    if et is not None and getattr(args, "et_algo3", False):
        zb = z(S - 1)
        diag["z_cosine"] = float(zb @ oracle / (np.linalg.norm(zb) * np.linalg.norm(oracle) + 1e-12))
        diag["z_norm_ratio"] = float(np.linalg.norm(zb) / (np.linalg.norm(oracle) + 1e-12))
    fname = "value_snapshots_episodes.npz" if episode else "value_snapshots.npz"
    snap = _SNAPSHOTS.setdefault((args.log_dir, fname), {"step": [], "episode": [], "values": [], "credit_trace": [],
                                                         "credit_agent": [], "credit_zb": [], "kernel_row": []})
    for k, v in [("step", t), ("episode", episode), ("values", values), ("credit_trace", credit_trace),
                 ("credit_agent", credit_agent), ("credit_zb", credit_zb), ("kernel_row", k_row)]:
        snap[k].append(v)
    snap.setdefault("reward_scale", []).append(scale)                          # reward normaliser at this moment
    snap.setdefault("raw_mean_q", []).append(float(np.mean(values)) / scale)   # network output before rescaling
    for k in ["trace_diversity", "feature_cosine", "feature_eff_rank", "kernel_spread", "head_share", "z_cosine", "z_norm_ratio"]:
        snap.setdefault("diag_" + k, []).append(diag.get(k, np.nan))
    np.savez_compressed(os.path.join(args.log_dir, fname), true=base.true_values(), chains=base.m, chain_len=n,
                        **{k: np.array(v) for k, v in snap.items()})


'''
HOOK_ANCHOR = "                log_dict.update(value_error_metrics(env, agent_state))\n"
EP_ANCHOR = "            episodes += 1\n"
EP_NEW = ("            if args.eval_values and args.snapshot_episodes and str(episodes) in args.snapshot_episodes.split(\",\"):\n"
          "                save_value_snapshot(env, agent_state, t, args, episode=episodes)\n")
HOOK_NEW = ("            if args.eval_values and args.snapshot_every and t % args.snapshot_every == 0:\n"
            "                save_value_snapshot(env, agent_state, t, args)\n")

ok = True
for f in FILES:
    s = open(f).read()
    if "raw_mean_q" in s:
        print(f"{f}: already patched, skipping"); continue
    if "def save_value_snapshot" in s:                        # earlier version: remove it, then re-apply
        a = s.index("_SNAPSHOTS = {}"); b = s.index(FUNC_ANCHOR)
        s = s[:a] + s[b:]
        s = s.replace(ARG_NEW, "").replace(ARG_V1, "").replace(HOOK_NEW, "").replace(EP_NEW, "")
        print(f"{f}: removing earlier snapshot patch")
    counts = {k: s.count(a) for k, a in [("args", ARG_ANCHOR), ("make_env", FUNC_ANCHOR), ("hook", HOOK_ANCHOR), ("episode", EP_ANCHOR)]}
    if any(c != 1 for c in counts.values()):
        print(f"{f}: NOT patched (run apply_eval_patch.py first?) anchors: {counts}"); ok = False; continue
    s = (s.replace(ARG_ANCHOR, ARG_ANCHOR + ARG_NEW).replace(FUNC_ANCHOR, FUNC_NEW + FUNC_ANCHOR)
          .replace(HOOK_ANCHOR, HOOK_ANCHOR + HOOK_NEW).replace(EP_ANCHOR, EP_ANCHOR + EP_NEW))
    open(f, "w").write(s)
    print(f"{f}: patched")
sys.exit(0 if ok else 1)
