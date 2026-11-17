"""Adds --q_lambda to algorithms/qrc.py and algorithms/qrc-et-algo3.py (R41, Peter's suggestion).

  --q_lambda   online Watkins Q(lambda): the Q update becomes  delta * trace  (QRC's gradient-correction terms that
               use the h network are dropped). With qrc-et-algo3.py the trace is QRC-ET's expected trace, so this gives
               Q(lambda) with expected traces. Everything else (traces cut on non-greedy actions, reward scaling,
               network, epsilon schedule) is unchanged. The h network still trains but no longer affects Q.

Run from the repo root AFTER apply_r38_r39_patch.py:  python apply_qlambda_patch.py   (safe to run twice)
"""
import sys

ARG_ANCHOR = "    freeze_body: bool = False\n"
ARG_NEW = ('    q_lambda: bool = False\n'
           '    """Online Watkins Q(lambda): drop the gradient-correction terms (Q update = delta * trace)."""\n')
FREEZE = "    if config.freeze_body:\n        q_update = mask_body(q_update)\n"
TRACE = {"algorithms/qrc.py": "grad_q_trace_t", "algorithms/qrc-et-algo3.py": "grad_q_trace_for_update"}

ok = True
for f, trace in TRACE.items():
    s = open(f).read()
    if "q_lambda: bool" in s:
        print(f"{f}: already patched, skipping"); continue
    if s.count(ARG_ANCHOR) != 1 or s.count(FREEZE) != 1 or f"tree.scale(td_error, {trace})" not in s:
        print(f"{f}: NOT patched (run apply_r38_r39_patch.py first?)"); ok = False; continue
    new = (f"    if config.q_lambda:  # online Watkins Q(lambda): no gradient correction\n"
           f"        q_update = tree.scale(td_error, {trace})\n")
    s = s.replace(ARG_ANCHOR, ARG_NEW + ARG_ANCHOR).replace(FREEZE, new + FREEZE)
    open(f, "w").write(s)
    print(f"{f}: patched (--q_lambda)")
sys.exit(0 if ok else 1)
