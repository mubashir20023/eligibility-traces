"""R44 summary: QRC-ET against QRC on 10 fresh seeds (601-610), per game, at lambda 0.9, lambda 0.95 and best-vs-best.
Score = mean rolling-100 return over the last 10% of 5M steps. Diverged = Q value not finite or above 1e4.
Usage (from stream-rep-rl):  python summarize_r44.py
"""
import os, re
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

TOTAL = 5_000_000
GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]
TAGS = [("lam09", "both at lambda 0.9"), ("lam095", "both at lambda 0.95"), ("best", "best setting of each")]
pat = re.compile(r"^(\w+)-v1__R44_(lam09|lam095|best)_(qrc|et)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)__(\d+)$")
runs = defaultdict(dict)
for e in os.scandir("logs/MinAtar"):
    m = pat.match(e.name)
    if not m:
        continue
    try:
        with open(f"logs/MinAtar/{e.name}/training_data.csv") as f:
            lines = f.read().splitlines()
        head = lines[0].split(","); i, j, k = head.index("global_step"), head.index("avg_return"), head.index("q_values")
        rows = [ln.split(",") for ln in lines[1:] if ln]
        num = lambda x: float(x) if x not in ("", "None") else np.nan
        st, ret, q = (np.array([num(r[c]) for r in rows]) for c in (i, j, k))
    except (OSError, ValueError, IndexError):
        continue
    if not len(st) or st[-1] < 0.99 * TOTAL:
        continue
    g, tag, meth, lam, ratio, lr, seed = m.groups()
    setting = f"lam {lam}, lr {lr}" + (f", ratio {ratio}" if ratio else "")
    runs[(g, tag, meth)][int(seed)] = (float(np.nanmean(ret[st > 0.9 * TOTAL])),
                                       bool(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4), setting)

print("R44: QRC-ET minus QRC on 10 fresh seeds (601-610); * = 95% interval excludes 0; [d] = diverged runs of 10")
summary = defaultdict(list)
for tag, name in TAGS:
    print(f"\n=== {name}")
    print(f"  {'game':14s} {'QRC':>22s} {'QRC-ET':>22s}   QRC-ET - QRC")
    for g in GAMES:
        q, e = runs.get((g, tag, "qrc"), {}), runs.get((g, tag, "et"), {})
        common = sorted(set(q) & set(e))
        if len(common) < 2:
            print(f"  {g:14s} (QRC {len(q)} runs, QRC-ET {len(e)} runs finished)")
            continue
        qs = np.array([q[s][0] for s in common]); es = np.array([e[s][0] for s in common]); d = es - qs
        h = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.26) * d.std(ddof=1) / np.sqrt(len(d))
        sig = "*" if (d.mean() - h > 0 or d.mean() + h < 0) else " "
        nd = lambda r: sum(v[1] for v in r.values())
        print(f"  {g:14s} {qs.mean():8.2f} [{nd(q)}]  {es.mean():8.2f} [{nd(e)}] "
              f"   {d.mean():+7.2f}{sig} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]  ET higher on {int((d > 0).sum())}/{len(d)}  (n={len(d)})")
        print(f"  {'':14s} settings: QRC {next(iter(q.values()))[2]} | QRC-ET {next(iter(e.values()))[2]}")
        summary[tag].append(100 * d.mean() / max(abs(qs.mean()), 1e-9))
print("\nSUMMARY R44 (QRC-ET - QRC as % of QRC, per game in order " + ", ".join(GAMES) + ")")
for tag, name in TAGS:
    v = summary[tag]
    if v:
        print(f"  {name:22s}: mean {np.mean(v):+6.1f}%   (" + ", ".join(f"{x:+.0f}%" for x in v) + ")")
