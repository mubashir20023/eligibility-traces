"""R39 summary: QRC vs QRC-ET on a pre-trained representation (seeds 501-510), next to the from-scratch R37 result.
For each game x representation source x (frozen / unfrozen): both scores, ET - QRC paired over seeds (95% CI), ET higher on.
Usage (from stream-rep-rl):  python summarize_r39.py
"""
import csv, glob, re
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

TOTAL = 5_000_000


def read(prefix, pattern):
    out = defaultdict(dict)
    for p in glob.glob(f"logs/**/*-v1__{prefix}_*__*/training_data.csv", recursive=True):
        m = re.search(rf"(\w+)-v1__{prefix}_{pattern}__(\d+)/", p)
        rows = list(csv.DictReader(open(p))) if m else []
        if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
            continue
        st = np.array([float(r["global_step"]) for r in rows]); ret = np.array([float(r["avg_return"]) for r in rows])
        out[m.groups()[:-1]][int(m.groups()[-1])] = float(ret[st > 0.9 * TOTAL].mean())
    return out


def compare(e, q):
    common = sorted(set(e) & set(q))
    if len(common) < 2:
        return "–"
    d = np.array([e[s] - q[s] for s in common])
    h = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.26) * d.std(ddof=1) / np.sqrt(len(d))
    return f"{d.mean():+7.2f} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}]  ET higher on {int((d > 0).sum())}/{len(d)}"


R39 = read("R39", r"(qrc|et)_src(qrc|et)_(frz|unf)_lr[\de.-]+")
R37 = read("R37", r"(qrc|et)_best_lr[\de.-]+")
ms = lambda d: f"{np.mean(list(d.values())):7.2f}±{np.std(list(d.values()), ddof=1) / np.sqrt(len(d)):.2f} [{len(d)}]" if len(d) > 1 else "      –"
for game in sorted({k[0] for k in R39}):
    print(f"\n=== {game}   (final score; higher = better)")
    print(f"  {'representation':28s} {'QRC':>18s} {'QRC-ET':>18s}   ET - QRC (95% CI)")
    q, e = R37.get((game, "qrc"), {}), R37.get((game, "et"), {})
    print(f"  {'from scratch (R37)':28s} {ms(q):>18s} {ms(e):>18s}   {compare(e, q)}")
    for src in ("qrc", "et"):
        for mode in ("frz", "unf"):
            q, e = R39.get((game, "qrc", src, mode), {}), R39.get((game, "et", src, mode), {})
            label = f"{'QRC' if src == 'qrc' else 'QRC-ET'}'s features, {'frozen' if mode == 'frz' else 'unfrozen'}"
            print(f"  {label:28s} {ms(q):>18s} {ms(e):>18s}   {compare(e, q)}")
