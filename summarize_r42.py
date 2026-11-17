"""R42 summary: full lambda sweep on the multi-chain, QRC against QRC-ET with the same grids.

Protocol (as in the expected-traces paper, plus held-out seeds): for every task, each agent's lambda and learning rate
are chosen on the tuning seeds (101-110) and the value error is then reported on the test seeds (111-120).
Value error = RMSE of the predicted values over ALL states, in three windows:
    after ~1,024 episodes (the paper's measure) | mean over the whole run (120k steps) | mean over the last 20%
Agents:  QRC  |  ET   = QRC-ET with a constant theta step (0.1 per visit, the deep-network setting)
                 ETn  = QRC-ET with theta step 1/visits (running mean of the traces, the paper's tabular setting)
For every task and window the script prints
    the error of the three agents at every lambda (best learning rate at that lambda), and ET / ETn against QRC there
    best QRC (any lambda) against best ET and best ETn (any lambda), paired over the test seeds
    a check of the constant theta step: ET at lambda 0.9 with step 0.3, 0.1 and 0.03, against best QRC.
Negative % = QRC-ET has the lower error. * = 95% interval excludes 0. "!" after a learning rate = edge of the grid.
A run with a non-finite error counts as diverged: its setting cannot be chosen, and it is listed if it happens on a test seed.

Usage (from stream-rep-rl):  python summarize_r42.py          (R42: QRC and QRC-ET)
                             python summarize_r42.py R42q     (optional online Q(lambda) version)
                             add  --short  to print only the final summary tables
"""
import csv, glob, os, pickle, re, sys
from collections import defaultdict
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SHORT = "--short" in sys.argv
PREFIX = ARGS[0] if ARGS else "R42"
TOTAL = 120000
TUNE, TEST = set(range(101, 111)), set(range(111, 121))
WINDOWS = ["after ~1,024 episodes", "whole run", "last 20%"]
FAMILY = {"MultiChainSparse": "our sparse task, one-hot", "MultiChainPaper": "paper's task, one-hot (Fig. 4)",
          "MultiChainPaperFeat": "paper's task, shared features (Fig. 5)"}
ET_AGENTS = [("et", "ET "), ("etcnt", "ETn")]
N_ARMS = 21

# ---------- read the runs (cached, so a second call is fast) ----------
cache_file = f"{PREFIX.lower()}_cache.pkl"
cache = pickle.load(open(cache_file, "rb")) if os.path.exists(cache_file) else {}
pat = re.compile(rf"multichain_env:(MultiChain\w+)-(m(\d+)-n(\d+)(?:-p([\d.]+))?)-v0__{PREFIX}_(qrc|etcnt|etb03|etb003|et)_lam([\d.]+)_lr([\d.]+)__(\d+)/")


def load(p):
    """(steps, value error) of one finished run, or None. Reads only the two columns that are needed."""
    try:
        with open(p) as f:
            lines = f.read().splitlines()
        head = lines[0].split(",")
        i, j = head.index("global_step"), head.index("value_rmse")
        rows = [ln.split(",") for ln in lines[1:] if ln]
        st = np.array([float(r[i]) for r in rows], dtype=np.float32)
        v = np.array([float(r[j]) if r[j] not in ("", "None") else np.nan for r in rows], dtype=np.float32)
        return (st, v) if len(st) and st[-1] >= 0.99 * TOTAL else None
    except (OSError, ValueError, IndexError):
        return None


# one listing of logs/ (a glob checks every run folder one by one, which is very slow on /fred)
found = {}
for e in os.scandir("logs"):
    p = f"logs/{e.name}/training_data.csv"
    m = pat.search(p)
    if m:
        found[p] = m
todo = [p for p in found if p not in cache]
if todo:
    from concurrent.futures import ThreadPoolExecutor
    print(f"reading {len(todo)} new runs ({len(found) - len(todo)} already cached) ...", file=sys.stderr, flush=True)
    with ThreadPoolExecutor(16) as pool:
        for k, (p, r) in enumerate(zip(todo, pool.map(load, todo)), 1):
            if r is not None:
                cache[p] = r
            if k % 2000 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}", file=sys.stderr, flush=True)
            if k % 10000 == 0 or k == len(todo):
                pickle.dump(cache, open(cache_file, "wb"))
runs = {p: m for p, m in found.items() if p in cache}

err = defaultdict(lambda: defaultdict(dict))        # (task, window) -> (agent, lambda, lr) -> seed -> error
info = {}
for p, m in runs.items():
    env, task, M, N, P, agent, lam, lr, seed = m.groups()
    key = f"{env}-{task}"
    info[key] = (env, int(M), int(N), P)
    st, v = cache[p]
    centre = 1024 * (int(N) + 2)                    # every episode has n + 2 steps
    masks = [(st >= 0.8 * centre) & (st <= 1.2 * centre), st >= 0, st >= 0.8 * TOTAL]
    for w, mask in zip(WINDOWS, masks):
        x = v[mask]
        err[(key, w)][(agent, float(lam), float(lr))][int(seed)] = float(x.mean()) if (len(x) and np.all(np.isfinite(x))) else np.inf


def mean_on(d, seeds):
    x = [d[s] for s in seeds if s in d]
    return float(np.mean(x)) if len(x) >= max(2, len(seeds) // 2) else None


def choose(cells, agent, lams):
    """Best (lambda, lr) of one agent over the given lambdas, by mean error on the tuning seeds."""
    best = None
    for (ag, lam, lr), d in cells.items():
        if ag == agent and lam in lams:
            mu = mean_on(d, TUNE)
            if mu is not None and np.isfinite(mu) and (best is None or mu < best[0]):
                best = (mu, lam, lr)
    return best


def pair(e, q, full=True):
    """QRC-ET against QRC on the test seeds, as text."""
    common = sorted(s for s in TEST if s in e and s in q)
    ok = [s for s in common if np.isfinite(e[s]) and np.isfinite(q[s])]
    if len(ok) < 2:
        return "-"
    d = np.array([e[s] - q[s] for s in ok]); qm = np.mean([q[s] for s in ok])
    h = (stats.t.ppf(0.975, len(d) - 1) if stats else 2.26) * d.std(ddof=1) / np.sqrt(len(d))
    txt = f"{100 * d.mean() / qm:+6.1f}%{'*' if (d.mean() + h < 0 or d.mean() - h > 0) else ' '}"
    if full:
        txt += f" [{100 * (d.mean() - h) / qm:+6.1f}, {100 * (d.mean() + h) / qm:+6.1f}]  better on {int((d < 0).sum())}/{len(d)}"
        if len(ok) < len(common):
            txt += f"  ({len(common) - len(ok)} test seeds diverged)"
    return txt


def order(key):
    env, M, N, P = info[key]
    return (list(FAMILY).index(env) if env in FAMILY else 9, P or "", M)


def title(key):
    env, M, N, P = info[key]
    return f"{FAMILY.get(env, env)}, m={M}" + (f", p={P}" if P else "") + f", chain length {N}"


summary = defaultdict(list)
for key in sorted(info, key=order):
    n_runs = sum(len(d) for d in err[(key, "whole run")].values())
    n_lrs = len({k[2] for k in err[(key, "whole run")]})
    if not SHORT:
        print(f"\n=== {title(key)}   [{key}; {n_runs} finished runs of {N_ARMS * n_lrs * 20}]")
    for w in WINDOWS:
        cells = err[(key, w)]
        lams = sorted({k[1] for k in cells})
        lrs = sorted({k[2] for k in cells})
        edge = lambda lr: "!" if lr in (lrs[0], lrs[-1]) else " "
        test_mean = lambda b, ag: mean_on(cells[(ag, b[1], b[2])], TEST)
        fmt = lambda b, ag: (f"{test_mean(b, ag):.4f} (lr {b[2]:g}{edge(b[2])})" if b and test_mean(b, ag) is not None else "-")
        if not SHORT:
            print(f"  {w}:")
            print(f"    {'lambda':>6s}  {'QRC':>24s}  {'ET':>24s}  {'ETn':>24s}   {'ET vs QRC':>10s}  {'ETn vs QRC':>10s}   (same lambda)")
        same9 = {}
        for lam in lams:
            b = {ag: choose(cells, ag, [lam]) for ag in ("qrc", "et", "etcnt")}
            c = {ag: (pair(cells[(ag, b[ag][1], b[ag][2])], cells[("qrc", b["qrc"][1], b["qrc"][2])], full=False) if b[ag] and b["qrc"] else "")
                 for ag in ("et", "etcnt")}
            if lam == 0.9:
                same9 = c
            if not SHORT:
                print(f"    {lam:>6g}  {fmt(b['qrc'], 'qrc'):>24s}  {fmt(b['et'], 'et'):>24s}  {fmt(b['etcnt'], 'etcnt'):>24s}   {c['et']:>10s}  {c['etcnt']:>10s}")
        bq = choose(cells, "qrc", lams)
        if not bq:
            continue
        lines = [f"best QRC lam {bq[1]:g} lr {bq[2]:g}{edge(bq[2])} error {test_mean(bq, 'qrc'):.4f}"]
        for ag, name in ET_AGENTS:
            be = choose(cells, ag, lams)
            if be:
                lines.append(f"best {name} lam {be[1]:g} lr {be[2]:g}{edge(be[2])} error {test_mean(be, ag):.4f} | vs best QRC "
                             f"{pair(cells[(ag, be[1], be[2])], cells[('qrc', bq[1], bq[2])])} | at lambda 0.9: {same9.get(ag, '') or '-'}")
        steps = []                                      # constant theta step check: ET at lambda 0.9 against the best QRC
        for ag, beta in (("etb03", "0.3"), ("et", "0.1"), ("etb003", "0.03")):
            b = choose(cells, ag, [0.9])
            if b:
                steps.append(f"{beta}: {test_mean(b, ag):.4f} (lr {b[2]:g}{edge(b[2])}) "
                             f"{pair(cells[(ag, b[1], b[2])], cells[('qrc', bq[1], bq[2])], full=False)}")
        if steps:
            lines.append("constant theta step, ET at lambda 0.9, vs best QRC:   " + "   ".join(steps))
        if not SHORT:
            for i, line in enumerate(lines):
                print(f"    {'BEST' if i == 0 else '    '}  {line}")
        summary[w].append((title(key), lines))

print(f"\nSUMMARY {PREFIX}: QRC-ET against QRC on the test seeds (negative = QRC-ET has the lower value error; * = significant;")
print("  ! = chosen learning rate is at the edge of the grid).  ET = constant theta step 0.1, ETn = theta step 1/visits.")
for w in WINDOWS:
    print(f"\n  {w}")
    for t, lines in summary[w]:
        print(f"    {t}")
        for line in lines:
            print(f"        {line}")
