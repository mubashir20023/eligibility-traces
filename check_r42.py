"""Checks before submitting R42 / R43. Run from stream-rep-rl:  python check_r42.py   (should end with "ALL OK")."""
import gymnasium as gym
import numpy as np
import multichain_env  # noqa: F401

ok = True
for eid, states in (("MultiChainSparse-m128-n4-p0.2-v0", 514), ("MultiChainPaper-m32-n4-v0", 130), ("MultiChainPaperFeat-m128-n16-v0", 2050),
                    ("MultiChainCode-m32-n4-p0.2-D16-k3-v0", 130)):
    e = gym.make("multichain_env:" + eid).unwrapped
    X = e.all_observations().reshape(e.n_states, -1)
    good = e.n_states == states and len({tuple(r) for r in X}) == states and np.all(np.isfinite(e.true_values()))
    print(("ok   " if good else "FAIL ") + eid)
    ok &= bool(good)

src = open("algorithms/qrc-et-algo3.py").read()
zero, split = src.find("if config.et_zero_theta:"), src.find("if config.et_split:  # z(S_t)")
good = "et_zero_theta: bool" in src and 0 < zero < split
print(("ok   " if good else "FAIL ") + "qrc-et-algo3.py: --et_zero_theta comes before the split step (theta-off control)")
ok &= good
good = "et_count_step: float" in src and 'self.param("count"' in src
print(("ok   " if good else "FAIL ") + "qrc-et-algo3.py: --et_count_step present (run: python apply_count_step_patch.py)")
ok &= good
for f in ("algorithms/qrc.py", "algorithms/qrc-et-algo3.py"):
    good = "q_lambda: bool" in open(f).read()
    print(("ok   " if good else "FAIL ") + f + ": --q_lambda option present")
    ok &= good
print("ALL OK" if ok else "SOMETHING IS MISSING - do not submit yet")
