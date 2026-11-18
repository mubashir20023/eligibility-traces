"""R44 stage 0: pick the settings to confirm on 10 fresh seeds, from the R36 + R43 tuning runs (seeds 301-303).
For every game: QRC and QRC-ET (split form, eta 0.3) at lambda 0.9 and at lambda 0.95 (best learning rate at that lambda,
QRC-ET at theta/Q ratio 100), and each method's best setting over everything tuned (any lambda; QRC-ET any ratio).
Writes r44_best.txt (one line per configuration: GAME METHOD LAMBDA RATIO LR TAG), 30 lines.
Usage (from stream-rep-rl):  python prepare_r44.py
"""
import os, re
from collections import defaultdict
import numpy as np

TOTAL = 5_000_000
GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]
pat = re.compile(r"^(\w+)-v1__(R36|R43)_(qrc|et)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)__(\d+)$")
score = defaultdict(dict)
for e in os.scandir("logs/MinAtar"):
    m = pat.match(e.name)
    if not m:
        continue
    try:
        with open(f"logs/MinAtar/{e.name}/training_data.csv") as f:
            lines = f.read().splitlines()
        head = lines[0].split(","); i, j = head.index("global_step"), head.index("avg_return")
        rows = [ln.split(",") for ln in lines[1:] if ln]
        st = np.array([float(r[i]) for r in rows]); ret = np.array([float(r[j]) if r[j] not in ("", "None") else np.nan for r in rows])
    except (OSError, ValueError, IndexError):
        continue
    if not len(st) or st[-1] < 0.99 * TOTAL:
        continue
    g, _, meth, lam, ratio, lr, seed = m.groups()
    score[(g, meth, float(lam), ratio or "0", lr)][int(seed)] = float(np.nanmean(ret[st > 0.9 * TOTAL]))


def best(g, meth, lam=None, ratio=None):
    cand = [(np.mean(list(d.values())), k) for k, d in score.items()
            if k[0] == g and k[1] == meth and (lam is None or k[2] == lam) and (ratio is None or k[3] == ratio) and len(d) >= 2]
    return max(cand)[1] if cand else None


out = []
for g in GAMES:
    for lam, tag in ((0.9, "lam09"), (0.95, "lam095")):
        for meth in ("qrc", "et"):
            k = best(g, meth, lam, None if meth == "qrc" else "100")
            if k:
                out.append((g, meth, f"{k[2]:g}", k[3], k[4], tag))
    for meth in ("qrc", "et"):
        k = best(g, meth)
        if k:
            out.append((g, meth, f"{k[2]:g}", k[3], k[4], "best"))
with open("r44_best.txt", "w") as f:
    for row in out:
        f.write(" ".join(row) + "\n")
for row in out:
    print(" ".join(row))
print(f"\n{len(out)} configurations written to r44_best.txt (should be 30). Then:  sbatch R44_minatar_confirm.sh")
