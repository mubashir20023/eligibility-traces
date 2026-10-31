"""Adds --et_split to algorithms/qrc-et-algo3.py (stable form of the no-lag fix, R35).

  --et_split   Theta learns only the PAST part of the trace,  z_past(s) ~ E[gamma*lambda*e_{t-1} | S_t = s],
               (Theta target = raw trace - grad q(S_t)), and the head update always adds the exact current gradient:
                   update direction = grad q(S_t) + z_past(S_t)        (eta = 0)
               With one-hot features this equals the no-lag expected trace z(S_t) in expectation (R33/R34a winner).
               With conv features the current-gradient term stays exact, as in the stable lagged code, so a stale or
               badly scaled Theta can no longer turn the update into a non-descent direction (R34c divergence).
               Use together with --no-et_algo3_lag (eta mixing then gives grad q + (1-eta) z_past + eta*gl*y_{t-1}).

Run from the repo root AFTER apply_eval_patch.py, apply_snapshot_patch.py and apply_linear_patch.py:
    python apply_split_patch.py          (safe to run twice; only qrc-et-algo3.py changes, default behaviour unchanged)
"""
import sys

F = "algorithms/qrc-et-algo3.py"
REPS = [
    ("    et_algo3_lag: bool = True\n",
     '    et_split: bool = False\n'
     '    """R35. Theta predicts only the past part of the trace; the head update adds the exact current gradient."""\n'
     "    et_algo3_lag: bool = True\n"),
    ('            tgt = raw_head_trace_full if config.et_algo3_target == "raw" else jax.lax.stop_gradient(e_w_t)\n',
     '            tgt = raw_head_trace_full if config.et_algo3_target == "raw" else jax.lax.stop_gradient(e_w_t)\n'
     "            if config.et_split:  # past part only: remove this step's gradient from the target\n"
     "                tgt = tgt - jax.lax.stop_gradient(head_grad_t)\n"),
    ("    eta = config.et_eta\n",
     "    if config.et_split:  # z(S_t) = exact current gradient + predicted past part\n"
     "        z_full = z_full + head_grad_t\n"
     "    eta = config.et_eta\n"),
    # snapshot / diagnostics: rebuild the full expected trace the same way
    ("        z = lambda s: np.asarray(et.apply_fn(et.params, jnp.asarray(X[s])))[0]\n",
     "        z = lambda s: np.asarray(et.apply_fn(et.params, jnp.asarray(X[s])))[0] + (X[s] if getattr(args, \"et_split\", False) else 0)\n"),
]

s = open(F).read()
if "et_split: bool" in s:
    print(f"{F}: already patched, skipping"); sys.exit(0)
counts = [s.count(a) for a, _ in REPS]
if any(c != 1 for c in counts):
    print(f"{F}: NOT patched, anchors found {counts} (expected all 1; run the eval/snapshot/linear patches first?)"); sys.exit(1)
for a, b in REPS:
    s = s.replace(a, b)
open(F, "w").write(s)
print(f"{F}: patched (--et_split added)")
