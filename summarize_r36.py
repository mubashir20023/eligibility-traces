"""R36 summary: equal-budget tuning on five MinAtar games, and the stage-2 list for R37.

Score = mean rolling-100 return over the last 10% of 5M steps, averaged over the 3 tuning seeds (301-303).
For each game prints the top configurations of each method, the effect of the Theta/Q rate ratio for QRC-ET,
and divergence counts; writes r36_best.txt (best QRC and best QRC-ET configuration per game) for R37.
Usage (from stream-rep-rl):  python summarize_r36.py
"""
import csv, glob, re
from collections import defaultdict
import numpy as np

TOTAL = 5_000_000
score, div = defaultdict(dict), defaultdict(int)
for p in glob.glob("logs/**/*-v1__R36_*__*/training_data.csv", recursive=True):
    m = re.search(r"(\w+)-v1__R36_(.+)__(\d+)/", p)
    rows = list(csv.DictReader(open(p))) if m else []
    if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
        continue
    st = np.array([float(r["global_step"]) for r in rows]); ret = np.array([float(r["avg_return"]) for r in rows])
    q = np.array([float(r["q_values"]) for r in rows])
    key = (m.group(1), m.group(2))
    score[key][int(m.group(3))] = float(ret[st > 0.9 * TOTAL].mean())
    div[key] += int(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4)


def parse(cfg):
    m = re.match(r"(qrc|et)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)$", cfg)
    return m.group(1), m.group(2), m.group(3), m.group(4)


best_lines = []
for game in sorted({k[0] for k in score}):
    print(f"\n=== {game}   (mean final score over tuning seeds [n, diverged])")
    best = {}
    for method in ("qrc", "et"):
        cfgs = [(np.mean(list(d.values())), c, len(d)) for (g, c), d in score.items() if g == game and parse(c)[0] == method]
        cfgs.sort(reverse=True)
        n_cfg = len(cfgs)
        print(f"  {'QRC' if method == 'qrc' else 'QRC-ET'}: {n_cfg} configurations complete, "
              f"{sum(div[(game, c)] for _, c, _ in cfgs)} diverged runs. Top 5:")
        for mu, c, n in cfgs[:5]:
            print(f"     {mu:7.2f}  {c}  [{n},{div[(game, c)]}]")
        if cfgs:
            best[method] = cfgs[0]
    ratio = defaultdict(list)
    for (g, c), d in score.items():
        if g == game and parse(c)[0] == "et":
            ratio[parse(c)[2]].append(np.mean(list(d.values())))
    if ratio:
        print("  QRC-ET by Theta/Q ratio (best / median over lr and lambda): " +
              ",  ".join(f"r={r}: {max(v):.1f} / {np.median(v):.1f}" for r, v in sorted(ratio.items(), key=lambda x: int(x[0]))))
    if "qrc" in best and "et" in best:
        print(f"  best QRC-ET - best QRC (tuning seeds, optimistic for both): {best['et'][0] - best['qrc'][0]:+.2f}")
        for method in ("qrc", "et"):
            kind, lam, r, lr = parse(best[method][1])
            best_lines.append(f"{game} {method} {lam} {r or '-'} {lr}")
with open("r36_best.txt", "w") as f:
    f.write("\n".join(best_lines) + "\n")
print(f"\nWrote r36_best.txt ({len(best_lines)} lines: game method lambda ratio lr) for R37_minatar_final.sh")
