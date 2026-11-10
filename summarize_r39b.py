"""R39b summary: QRC vs QRC-ET on frozen pre-trained conv layers with IDENTICAL settings (lr x lambda) for both.

Per game and representation source:
  * every setting: QRC score, QRC-ET score, ET - QRC paired over seeds 501-505 with 95% CI, marked + / - if significant
  * count of settings where QRC-ET is significantly better / worse, and the mean difference over all settings
  * best setting of each method
Usage (from stream-rep-rl):  python summarize_r39b.py
"""
import csv, glob, re
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

TOTAL = 5_000_000
S = defaultdict(dict)
for p in glob.glob("logs/**/*-v1__R39b_*__*/training_data.csv", recursive=True):
    m = re.search(r"(\w+)-v1__R39b_(qrc|et)_src(qrc|et)_lam([\d.]+)_lr([\de.-]+)__(\d+)/", p)
    rows = list(csv.DictReader(open(p))) if m else []
    if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
        continue
    st = np.array([float(r["global_step"]) for r in rows]); ret = np.array([float(r["avg_return"]) for r in rows])
    g, meth, src, lam, lr, seed = m.groups()
    S[(g, src, lam, lr, meth)][int(seed)] = float(ret[st > 0.9 * TOTAL].mean())

overall = []
for g in sorted({k[0] for k in S}):
    for src in ("qrc", "et"):
        cells = sorted({(k[2], k[3]) for k in S if k[0] == g and k[1] == src}, key=lambda x: (float(x[0]), float(x[1])))
        if not cells:
            continue
        print(f"\n=== {g}, frozen {'QRC' if src == 'qrc' else 'QRC-ET'} features   (final score; higher = better)")
        print(f"  {'lambda':>6} {'lr':>8} {'QRC':>9} {'QRC-ET':>9}   ET - QRC (95% CI)")
        better = worse = 0; diffs = []
        best = {"qrc": (-1e9, None), "et": (-1e9, None)}
        for lam, lr in cells:
            q, e = S.get((g, src, lam, lr, "qrc"), {}), S.get((g, src, lam, lr, "et"), {})
            for meth, d in (("qrc", q), ("et", e)):
                if d and np.mean(list(d.values())) > best[meth][0]:
                    best[meth] = (np.mean(list(d.values())), f"lambda {lam}, lr {lr}")
            common = sorted(set(q) & set(e))
            if len(common) < 2:
                print(f"  {lam:>6} {lr:>8}   (incomplete)"); continue
            d = np.array([e[s] - q[s] for s in common])
            h = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.78) * d.std(ddof=1) / np.sqrt(len(d))
            mark = "+" if d.mean() - h > 0 else ("-" if d.mean() + h < 0 else " ")
            better += mark == "+"; worse += mark == "-"; diffs.append(d.mean())
            print(f"  {lam:>6} {lr:>8} {np.mean([q[s] for s in common]):9.2f} {np.mean([e[s] for s in common]):9.2f}   "
                  f"{d.mean():+7.2f} [{d.mean() - h:+.2f}, {d.mean() + h:+.2f}] {mark}  (n={len(d)})")
        print(f"  -> QRC-ET significantly better in {better}, worse in {worse}, of {len(diffs)} settings; "
              f"mean ET - QRC over settings {np.mean(diffs):+.2f}")
        print(f"     best QRC {best['qrc'][0]:.2f} ({best['qrc'][1]}),  best QRC-ET {best['et'][0]:.2f} ({best['et'][1]})")
        overall.append(f"{g:14s} {src:4s} features: ET better {better}, worse {worse} of {len(diffs)}; "
                       f"mean diff {np.mean(diffs):+.2f}; best QRC {best['qrc'][0]:.1f} vs best QRC-ET {best['et'][0]:.1f}")
print("\nSUMMARY R39b (frozen features, identical settings; + / - = significant at 95%)")
for line in overall:
    print("  " + line)
