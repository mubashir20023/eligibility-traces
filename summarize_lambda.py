"""Same-lambda view of the MinAtar tuning runs, with the R43 controls.

The expected-traces paper compares QET(lambda) with Q(lambda) at ONE long trace (lambda = 0.95, Fig. 6). Our tuning jobs
compare the best setting of each method over all lambda. This script shows, for every game and every lambda, the best
learning rate of each agent at that SAME lambda:
    QRC          the base agent (Q(lambda) for R41)
    QRC-ET       split form, eta 0.3, theta/Q learning-rate ratio 100
    theta off    the same agent with theta's output set to 0 (R43 control: only the shorter head trace is left)
    Algorithm 3  the paper's Algorithm 3 form (one-step lag, theta trained on the mixed trace), eta 0.3, ratio 100
    Alg. 3 eta 0 the same with eta 0 (pure expected trace, the paper's Fig. 6 setting), lambda 0.95 and 0.8 only
Score = mean rolling-100 return over the last 10% of 5M steps, mean over the tuning seeds (301-303).
[d/n] = diverged runs / runs over the four learning rates (diverged: logged Q value not finite or above 1e4).

Usage (from stream-rep-rl):  python summarize_lambda.py R36 R43     (conv, QRC; R43 adds the missing cells)
                             python summarize_lambda.py R36         |  python summarize_lambda.py R41
                             python summarize_lambda.py R38mlp      |  python summarize_lambda.py R38lin2
"""
import csv, glob, os, pickle, re, sys
from collections import defaultdict
import numpy as np

PREFIXES = sys.argv[1:] or ["R36"]
PRE = "(?:" + "|".join(PREFIXES) + ")"
TOTAL = 5_000_000
ARMS = [("qrc", "QRC"), ("et", "QRC-ET"), ("zero", "theta off"), ("alg3", "Algorithm 3"), ("alg3e0", "Alg. 3 eta 0")]
score, div = defaultdict(dict), defaultdict(int)
PAT = re.compile(rf"(\w+)-v1__{PRE}_(qrc|et|zero|alg3e0|alg3)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)__(\d+)/")


def load(p):
    """(final score, diverged) of one finished run, or None. Reads only the three columns that are needed."""
    try:
        with open(p) as f:
            lines = f.read().splitlines()
        head = lines[0].split(",")
        i, j, k = head.index("global_step"), head.index("avg_return"), head.index("q_values")
        rows = [ln.split(",") for ln in lines[1:] if ln]
        num = lambda x: float(x) if x not in ("", "None") else np.nan
        st, ret, q = (np.array([num(r[c]) for r in rows]) for c in (i, j, k))
        if not len(st) or st[-1] < 0.99 * TOTAL:
            return None
        return float(ret[st > 0.9 * TOTAL].mean()), int(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4)
    except (OSError, ValueError, IndexError):
        return None


# one listing of logs/MinAtar (a recursive glob walks every run folder under logs/, which is very slow on /fred)
cache_file = "lambda_cache.pkl"
cache = pickle.load(open(cache_file, "rb")) if os.path.exists(cache_file) else {}
found = {}
for e in os.scandir("logs/MinAtar"):
    p = f"logs/MinAtar/{e.name}/training_data.csv"
    m = PAT.search(p)
    if m and (m.group(2) == "qrc" or m.group(4) == "100"):       # other theta ratios (R36: 30, 300) are left out
        found[p] = m
todo = [p for p in found if p not in cache]
if todo:
    from concurrent.futures import ThreadPoolExecutor
    print(f"reading {len(todo)} new runs ({len(found) - len(todo)} already cached) ...", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(8) as pool:
        for n, (p, r) in enumerate(zip(todo, pool.map(load, todo)), 1):
            if r is not None:
                cache[p] = r
            if n % 100 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", file=sys.stderr, flush=True)
    pickle.dump(cache, open(cache_file, "wb"))
for p, m in found.items():
    if p in cache:
        key = (m.group(1), m.group(2), float(m.group(3)), m.group(5))                  # game, arm, lambda, lr
        score[key][int(m.group(6))] = cache[p][0]
        div[key] += cache[p][1]


def best(game, arm, lam=None):
    """(mean score, lr, lambda, diverged runs, runs over all lr) of the best learning rate (and lambda, if not given)."""
    cand, nd, nr = [], 0, 0
    for (g, a, l, lr), d in score.items():
        if g == game and a == arm and (lam is None or l == lam):
            cand.append((float(np.mean(list(d.values()))), lr, l))
            nd += div[(g, a, l, lr)]; nr += len(d)
    return (max(cand) + (nd, nr)) if cand else None


games = sorted({k[0] for k in score})
lams = sorted({k[2] for k in score})
arms = [(a, n) for a, n in ARMS if any(k[1] == a for k in score)]
base = "Q(lambda)" if any(p.startswith("R41") for p in PREFIXES) else "QRC"
tag = "+".join(PREFIXES)
cell = lambda b: f"{b[0]:7.2f} @{b[1]} [{b[3]}/{b[4]}]" if b else "-"
agg, top = defaultdict(lambda: defaultdict(list)), defaultdict(list)
for g in games:
    ref = best(g, "qrc") or max(b for b in (best(g, a) for a, _ in arms) if b)      # best base score over all lambda
    ref = ref[0]
    print(f"\n=== {g}   ({tag}; best learning rate per cell; [diverged/runs over all learning rates])")
    print(f"  {'lambda':>6}" + "".join(f"  {(base if a == 'qrc' else n):>22}" for a, n in arms))
    for l in lams:
        b = {a: best(g, a, l) for a, _ in arms}
        print(f"  {l:>6g}" + "".join(f"  {cell(b[a]):>22}" for a, _ in arms))
        for a, _ in arms:
            if a != "qrc" and b[a] and b.get("qrc"):
                agg[(a, "qrc")][l].append((b[a][0] - b["qrc"][0]) / ref)
        if b.get("et") and b.get("zero"):
            agg[("et", "zero")][l].append((b["et"][0] - b["zero"][0]) / ref)
    line = f"  {'best':>6}"
    for a, _ in arms:
        b = best(g, a)
        line += f"  {(f'{b[0]:7.2f} @{b[1]} lam {b[2]:g}') if b else '-':>22}"
        if b and a != "qrc":
            top[a].append((b[0] - ref) / ref)
    print(line)

name = dict(ARMS); name["qrc"] = base
print(f"\nSUMMARY {tag}: difference at the SAME lambda, as % of the game's best {base} score; mean over games (per game)")
for (a, b), per_lam in agg.items():
    print(f"\n  {name[a]} minus {name[b]}")
    for l in sorted(per_lam):
        v = np.array(per_lam[l])
        print(f"    lambda {l:<5g}: {100 * v.mean():+6.1f}%   (" + ", ".join(f"{100 * x:+.0f}%" for x in v) + ")")
if top:
    print(f"\n  Best setting over all lambda and learning rates, minus best {base} (% of best {base}; mean over games (per game))")
    for a, v in top.items():
        v = np.array(v)
        print(f"    {name[a]:12s}: {100 * v.mean():+6.1f}%   (" + ", ".join(f"{100 * x:+.0f}%" for x in v) + ")")
print(f"Games in order: {', '.join(games)}. Three tuning seeds per cell: a screen, not a test.")
