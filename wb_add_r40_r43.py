"""Add two sections to the existing report, as the newest follow-ups (inserted right before the R38-R39 section):
    Latest    | R42-R43: full lambda sweeps on the multi-chain (with the paper's own tasks) and the missing MinAtar cells
    Follow-up | R40-R41: why? feature sharing (multi-hot codes) and online Q(lambda)
Same layout as before: compact tables + figures. Also writes RESULTS_R40_R43.md with the same tables.

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_add_r40_r43.py --dry-run     # tables + figures + RESULTS_R40_R43.md; changes nothing online
    ~/wbenv/bin/python wb_add_r40_r43.py               # log the figures run, back up the report, insert, save
Needs r42_cache.pkl (written by: python summarize_r42.py). The other runs are read once and cached in
wb_r40_r43_cache.pkl. Figures follow one colour per agent: QRC blue, QRC-ET red, QRC-ET 1/visits green,
Algorithm 3 form yellow, theta on state id violet, theta off grey (checked for colour-blind separation).
"""
import argparse, os, pickle, re, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

from wb_reorganize import md, block_text, REPORT_URL, GITHUB

TAG = "FOLLOWUP-R40-R43-FIGS"
FIGDIR = "report_figures"
NEXT_SECTION = "Non-Convolutional Networks and Pre-Trained Representations"     # R38-R39 heading we insert in front of
TITLE_NEW = "Full λ Sweeps, the Paper's Own Tasks and the Missing MinAtar Cells (R42, R43)"
TITLE_WHY = "Why: Feature Sharing and Online Q(λ) (R40, R41)"
GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]
TOTAL_MINATAR, TOTAL_MC = 5_000_000, 120_000
TUNE, TEST = set(range(101, 111)), set(range(111, 121))
WINDOWS = ["early", "whole", "late"]                      # after ~1,024 episodes | whole run | last 20%
FAMILIES = [(("MultiChainSparse", "0.2"), "Sparse reward p=0.2, one-hot"), (("MultiChainSparse", "0.9"), "Sparse reward p=0.9, one-hot"),
            (("MultiChainPaper", ""), "Paper's task, one-hot (Fig. 4)"), (("MultiChainPaperFeat", ""), "Paper's task, shared features (Fig. 5)")]
CHAINS = [8, 32, 128]
CODES = [(130, "one-hot", 0.000, 0.0), (64, "D=64", 0.048, 51.1), (32, "D=32", 0.096, 58.7), (16, "D=16", 0.190, 55.5),
         (10, "D=10", 0.499, 45.0)]                       # code length, label, feature overlap, best linear theta error (%)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
COL = dict(qrc="#1f77b4", et="#d62728", etn="#1baf7a", alg3="#eda100", off="#7f7f7f", oracle="#6250d6")
MARK = dict(qrc="o", et="s", etn="^", alg3="D", off="v", oracle="P")


# ------------------------------------------------------------------------------------------------ statistics
def tcrit(n):
    return float(stats.t.ppf(0.975, n - 1)) if stats else {2: 12.71, 3: 4.30, 5: 2.78, 10: 2.26}.get(n, 2.1)


def paired(a, b, seeds=None):
    """a against b over common seeds: mean difference with 95% interval, absolute and relative to mean(b) in %."""
    pool = seeds if seeds is not None else set(a) & set(b)
    common = sorted(s for s in pool if s in a and s in b and np.isfinite(a[s]) and np.isfinite(b[s]))
    if len(common) < 2:
        return None
    d = np.array([a[s] - b[s] for s in common])
    bm = float(np.mean([b[s] for s in common])) or 1e-12
    h = tcrit(len(d)) * d.std(ddof=1) / np.sqrt(len(d))
    return dict(n=len(d), diff=float(d.mean()), lo=float(d.mean() - h), hi=float(d.mean() + h), rel=100 * d.mean() / bm,
                rel_lo=100 * (d.mean() - h) / bm, rel_hi=100 * (d.mean() + h) / bm,
                sig=bool(d.mean() - h > 0 or d.mean() + h < 0), lower=int((d < 0).sum()), higher=int((d > 0).sum()))


def pct(c, digits=1):
    if c is None:
        return "–"
    s = f"{c['rel']:+.{digits}f}%"
    return f"**{s}**" if c["sig"] else s


def absdiff(c):
    if c is None:
        return "–"
    s = f"{c['diff']:+.1f} [{c['lo']:+.1f}, {c['hi']:+.1f}]"
    return f"**{s}**" if c["sig"] else s


# ------------------------------------------------------------------------------------------------ reading
def read_cols(path, cols):
    with open(path) as f:
        lines = f.read().splitlines()
    head = lines[0].split(",")
    idx = [head.index(c) for c in cols]
    rows = [ln.split(",") for ln in lines[1:] if ln]
    num = lambda x: float(x) if x not in ("", "None") else np.nan
    return [np.array([num(r[i]) for r in rows]) for i in idx]


def load_minatar(path):
    """(final score, diverged) of one finished MinAtar run, or None."""
    try:
        st, ret, q = read_cols(path, ["global_step", "avg_return", "q_values"])
        if not len(st) or st[-1] < 0.99 * TOTAL_MINATAR:
            return None
        return float(ret[st > 0.9 * TOTAL_MINATAR].mean()), bool(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4)
    except (OSError, ValueError, IndexError):
        return None


def load_code(path):
    """Mean value error over the run of one finished multi-chain run (as summarize_r40.py), or None."""
    try:
        st, v = read_cols(path, ["global_step", "value_rmse"])
        if not len(st) or st[-1] < 0.99 * TOTAL_MC:
            return None
        return float(np.mean(np.where(np.isfinite(v), v, 10.0)))
    except (OSError, ValueError, IndexError):
        return None


def read_runs(cache_file="wb_r40_r43_cache.pkl"):
    """MinAtar: {prefix: {(game, config): {seed: dict(score, diverged)}}};  codes: {(p, D, agent, lr): {seed: error}}."""
    cache = pickle.load(open(cache_file, "rb")) if os.path.exists(cache_file) else {}
    pm = re.compile(r"^(\w+)-v1__(R36|R37|R38mlpF|R38lin2F|R41F|R41|R43)_(.+)__(\d+)$")
    pc = re.compile(r"^multichain_env:MultiChainCode(?:Id)?-m32-n4-p([\d.]+)-D(\d+)-k\d+-v0__(?:R40c|R40b|R40)_(.+)_lr([\d.]+)__(\d+)$")
    names_m = [e.name for e in os.scandir("logs/MinAtar")] if os.path.isdir("logs/MinAtar") else []     # one listing each:
    names_c = [e.name for e in os.scandir("logs") if e.name.startswith("multichain_env:MultiChainCode")]  # no slow globs
    jobs = [(f"logs/MinAtar/{n}/training_data.csv", load_minatar) for n in names_m if pm.match(n)]
    jobs += [(f"logs/{n}/training_data.csv", load_code) for n in names_c if pc.match(n)]
    jobs = [(p, f) for p, f in jobs if p not in cache]
    if jobs:
        print(f"reading {len(jobs)} runs (cached afterwards) ...", file=sys.stderr, flush=True)
        with ThreadPoolExecutor(8) as pool:
            for k, ((p, _), r) in enumerate(zip(jobs, pool.map(lambda pf: pf[1](pf[0]), jobs)), 1):
                if r is not None:
                    cache[p] = r
                if k % 250 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)}", file=sys.stderr, flush=True)
        pickle.dump(cache, open(cache_file, "wb"))
    minatar, codes = defaultdict(lambda: defaultdict(dict)), defaultdict(dict)
    for n in names_m:
        m, p = pm.match(n), f"logs/MinAtar/{n}/training_data.csv"
        if m and p in cache:
            minatar[m.group(2)][(m.group(1), m.group(3))][int(m.group(4))] = dict(score=cache[p][0], diverged=cache[p][1])
    for n in names_c:
        m, p = pc.match(n), f"logs/{n}/training_data.csv"
        if m and p in cache:
            codes[(m.group(1), int(m.group(2)), m.group(3), m.group(4))][int(m.group(5))] = cache[p]
    return minatar, codes


# ------------------------------------------------------------------------------------------------ R42 (multi-chain sweep)
def analyse_r42(cache_file="r42_cache.pkl", prefix="R42"):
    """Same protocol as summarize_r42.py: lambda and learning rate chosen on seeds 101-110, reported on 111-120."""
    if not os.path.exists(cache_file):
        raise SystemExit(f"{cache_file} not found: run  python summarize_r42.py {prefix}  first.")
    cache = pickle.load(open(cache_file, "rb"))
    pat = re.compile(r"multichain_env:(MultiChain\w+)-m(\d+)-n(\d+)(?:-p([\d.]+))?-v0__" + prefix + r"_(qrc|etcnt|etb03|etb003|et)_lam([\d.]+)_lr([\d.]+)__(\d+)/")
    err = defaultdict(lambda: defaultdict(dict))            # (task, window) -> (agent, lambda, lr) -> seed -> error
    for p, (st, v) in cache.items():
        m = pat.search(p)
        if not m:
            continue
        env, M, N, P, ag, lam, lr, seed = m.groups()
        centre = 1024 * (int(N) + 2)
        masks = [(st >= 0.8 * centre) & (st <= 1.2 * centre), st >= 0, st >= 0.8 * TOTAL_MC]
        for w, mask in zip(WINDOWS, masks):
            x = v[mask]
            err[((env, P or "", int(M)), w)][(ag, float(lam), float(lr))][int(seed)] = (
                float(x.mean()) if (len(x) and np.all(np.isfinite(x))) else np.inf)

    def mean_on(d, seeds):
        x = [d[s] for s in seeds if s in d]
        return float(np.mean(x)) if len(x) >= max(2, len(seeds) // 2) else None

    def choose(cells, agent, lams):
        best = None
        for (ag, lam, lr), d in cells.items():
            if ag == agent and lam in lams:
                mu = mean_on(d, TUNE)
                if mu is not None and np.isfinite(mu) and (best is None or mu < best[0]):
                    best = (mu, lam, lr)
        return best

    out = {}
    for (task, w), cells in err.items():
        lams = sorted({k[1] for k in cells})
        r = dict(lams=lams, curve=defaultdict(dict))
        bq = choose(cells, "qrc", lams)
        if not bq:
            continue
        qd = cells[("qrc", bq[1], bq[2])]
        r["qrc"] = dict(lam=bq[1], lr=bq[2], mean=mean_on(qd, TEST), seeds=qd)
        for ag in ("et", "etcnt"):
            be = choose(cells, ag, lams)
            if be:
                r[ag] = dict(lam=be[1], lr=be[2], mean=mean_on(cells[(ag, be[1], be[2])], TEST), cmp=paired(cells[(ag, be[1], be[2])], qd, TEST),
                             seeds=cells[(ag, be[1], be[2])])
        for ag in ("etb03", "et", "etb003", "etcnt"):          # theta step check at lambda 0.9, against the best QRC
            b9 = choose(cells, ag, [0.9])
            r["step_" + ag] = paired(cells[(ag, b9[1], b9[2])], qd, TEST) if b9 else None
        q9 = choose(cells, "qrc", [0.9])
        for ag in ("et", "etcnt"):                             # both agents at lambda 0.9
            b9 = choose(cells, ag, [0.9])
            r["same9_" + ag] = paired(cells[(ag, b9[1], b9[2])], cells[("qrc", q9[1], q9[2])], TEST) if (b9 and q9) else None
        for ag in ("qrc", "et", "etcnt"):                      # error against lambda (best learning rate at each lambda)
            for lam in lams:
                b = choose(cells, ag, [lam])
                if b:
                    r["curve"][ag][lam] = mean_on(cells[(ag, b[1], b[2])], TEST)
        out[(task, w)] = r
    return out, len(cache)


# ------------------------------------------------------------------------------------------------ figures
def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel(xl, color=MUTED, fontsize=9)
    ax.set_ylabel(yl, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)


def draw(ax, xs, key, label, y, lo=None, hi=None, dashed=False):
    """One series: 2px line, markers with a white ring, optional 95% interval whiskers."""
    y = np.array([np.nan if v is None else v for v in y], float)
    y[~np.isfinite(y)] = np.nan
    if lo is not None:
        lo = np.array([np.nan if v is None else v for v in lo], float)
        hi = np.array([np.nan if v is None else v for v in hi], float)
        ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
        ax.errorbar(np.array(xs)[ok], y[ok], yerr=[(y - lo)[ok], (hi - y)[ok]], fmt="none", ecolor=COL[key], elinewidth=1.3,
                    capsize=0, zorder=2)
    ax.plot(xs, y, color=COL[key], lw=2, ls="--" if dashed else "-", marker=MARK[key], ms=7.5, mec="white", mew=1.3,
            label=label, zorder=3, solid_capstyle="round")


def finish(fig, axes, path, top=0.90, handlelength=2.2):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    if len(labels) > 1:                                        # one legend for the whole figure; a single series needs none
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.005, 1.0), ncol=len(labels), frameon=False, fontsize=9,
                   labelcolor=INK, handlelength=handlelength, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, top if len(labels) > 1 else 1))
    fig.savefig(path, dpi=150, facecolor="white")
    return path


def make_figures(R42, L, CODE, FOREST):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    F = {}

    # F1: QRC-ET against the best QRC, by number of chains (whole run), one panel per task family
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.2), sharey=True)
    for ax, (fam, name) in zip(axes.ravel(), FAMILIES):
        xs = np.arange(len(CHAINS))
        for ag, lab in (("et", "QRC-ET, constant theta step (0.1 per visit)"), ("etcnt", "QRC-ET, theta step 1/visits")):
            c = [R42.get(((fam[0], fam[1], m), "whole"), {}).get(ag, {}).get("cmp") for m in CHAINS]
            draw(ax, xs, "et" if ag == "et" else "etn", lab, [x and x["rel"] for x in c], [x and x["rel_lo"] for x in c], [x and x["rel_hi"] for x in c])
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels([str(m) for m in CHAINS]); ax.set_xlim(-0.35, len(CHAINS) - 0.65)
        style(ax, name, "number of chains (m)", "value error against best QRC (%)\nbelow 0 = QRC-ET better")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    F["r42_advantage"] = finish(fig, axes, os.path.join(FIGDIR, "r42_advantage.png"), top=0.94); plt.close(fig)

    # F2: value error against lambda at m = 32 (the paper's task, one-hot and shared features)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    for ax, (fam, name) in zip(axes, FAMILIES[2:]):
        r = R42.get(((fam[0], fam[1], 32), "whole"))
        if r:
            xs = np.arange(len(r["lams"]))
            for ag, key, lab in (("qrc", "qrc", "QRC"), ("et", "et", "QRC-ET, constant theta step"), ("etcnt", "etn", "QRC-ET, theta step 1/visits")):
                draw(ax, xs, key, lab, [r["curve"][ag].get(l) for l in r["lams"]])
            ax.set_xticks(xs); ax.set_xticklabels([f"{l:g}" for l in r["lams"]])
            ax.set_ylim(bottom=0)
        style(ax, name + ", m = 32", "trace length λ (best learning rate at each λ)", "value error, whole run (test seeds)")
    F["r42_lambda"] = finish(fig, axes, os.path.join(FIGDIR, "r42_lambda.png")); plt.close(fig)

    # F3: MinAtar, score and divergence against lambda (mean of five games)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    xs = np.arange(len(L["lams"]))
    for arm, key, lab, dashed in (("qrc", "qrc", "QRC", False), ("et", "et", "QRC-ET (split form)", False),
                                  ("alg3", "alg3", "QRC-ET (Algorithm 3 form)", False), ("zero", "off", "QRC-ET, theta off", True)):
        draw(axes[0], xs, key, lab, [L["rel"][arm].get(l) for l in L["lams"]], dashed=dashed)
        draw(axes[1], xs, key, lab, [L["div"][arm].get(l) for l in L["lams"]], dashed=dashed)
    for ax, t, yl in ((axes[0], "MinAtar: best score at each λ (mean of 5 games)", "score ÷ best QRC score of the game"),
                      (axes[1], "MinAtar: runs with diverged Q values", "diverged runs (%)")):
        ax.set_xticks(xs); ax.set_xticklabels([f"{l:g}" for l in L["lams"]])
        ax.set_ylim(bottom=-0.04 * ax.get_ylim()[1])
        style(ax, t, "trace length λ", yl)
    F["r43_lambda"] = finish(fig, axes, os.path.join(FIGDIR, "r43_lambda.png")); plt.close(fig)

    # F4: feature sharing (R40): QRC-ET against best QRC for every code
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), sharey=True)
    xs = np.arange(len(CODES))
    for ax, p in zip(axes, ("0.2", "0.9")):
        for fam, key, lab in (("et", "et", "QRC-ET, theta on the shared features"), ("oracle", "oracle", "QRC-ET, theta on a one-hot state id")):
            c = [CODE.get((p, D), {}).get(fam) for D, *_ in CODES]
            draw(ax, xs, key, lab, [x and x["rel"] for x in c], [x and x["rel_lo"] for x in c], [x and x["rel_hi"] for x in c])
        ax.axhline(0, color=MUTED, lw=1, zorder=1)
        ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in CODES])
        style(ax, f"Sparse multi-chain, reward p = {p}", "feature code (more sharing to the right)", "value error against best QRC (%)\nbelow 0 = QRC-ET better")
    axes[1].set_ylabel("")
    F["r40_sharing"] = finish(fig, axes, os.path.join(FIGDIR, "r40_sharing_report.png")); plt.close(fig)

    # F5: every MinAtar final comparison (10 fresh seeds): QRC-ET minus baseline with 95% interval
    panels = [k for k in ("R37", "R41F", "R38mlpF", "R38lin2F") if FOREST.get(k)]
    if panels:
        names = dict(R37="Conv network, QRC", R41F="Conv network, online Q(λ)", R38mlpF="MLP, QRC", R38lin2F="Linear on pixels, QRC")
        fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.6), sharey=True)
        axes = np.atleast_1d(axes)
        ys = np.arange(len(GAMES))[::-1]
        for ax, k in zip(axes, panels):
            c = [FOREST[k].get(g) for g in GAMES]
            x = np.array([np.nan if v is None else v["rel"] for v in c]); lo = np.array([np.nan if v is None else v["rel_lo"] for v in c])
            hi = np.array([np.nan if v is None else v["rel_hi"] for v in c]); ok = np.isfinite(x)
            ax.errorbar(x[ok], ys[ok], xerr=[(x - lo)[ok], (hi - x)[ok]], fmt="none", ecolor=COL["et"], elinewidth=1.6, capsize=0, zorder=2)
            ax.plot(x[ok], ys[ok], ls="none", marker="s", ms=7.5, color=COL["et"], mec="white", mew=1.3, zorder=3)
            ax.axvline(0, color=MUTED, lw=1, zorder=1)
            ax.set_yticks(ys); ax.set_yticklabels(GAMES); ax.set_ylim(-0.6, len(GAMES) - 0.4)
            style(ax, names[k], "QRC-ET − baseline (% of baseline)", "")
        F["minatar_forest"] = finish(fig, axes, os.path.join(FIGDIR, "minatar_forest.png")); plt.close(fig)
    return F


# ------------------------------------------------------------------------------------------------ tables
def build(minatar, codes, R42):
    T, S = {}, {}
    score = lambda d: {s: v["score"] for s, v in d.items()}
    mean = lambda d: float(np.mean([v["score"] for v in d.values()]))

    # ---------- R42
    rows, steps, onehot_whole, feat_whole = [], [], [], []
    for fam, name in FAMILIES:
        for m in CHAINS:
            cell = lambda ag, w: (R42.get(((fam[0], fam[1], m), w), {}).get(ag) or {}).get("cmp")
            q = R42.get(((fam[0], fam[1], m), "whole"), {}).get("qrc")
            rows.append([name if m == CHAINS[0] else "", m, f"{q['lam']:g}" if q else "–"] + [pct(cell("et", w)) for w in WINDOWS]
                        + [pct(cell("etcnt", w)) for w in WINDOWS])
            e = R42.get(((fam[0], fam[1], m), "early"), {})
            steps.append([name if m == CHAINS[0] else "", m] + [pct(e.get("step_" + ag)) for ag in ("etb03", "et", "etb003", "etcnt")])
            (feat_whole if fam[0].endswith("Feat") else onehot_whole).append((cell("et", "whole"), cell("etcnt", "whole")))
    T["r42_main"] = md(["Task", "Chains", "Best QRC λ", "ET const.: early", "whole run", "last 20%", "ET 1/visits: early", "whole run", "last 20%"], rows)
    T["r42_step"] = md(["Task", "Chains", "θ step 0.3", "0.1", "0.03", "1/visits"], steps)
    lam_rows = []
    r1, r2 = R42.get((("MultiChainPaper", "", 32), "whole")), R42.get((("MultiChainPaperFeat", "", 32), "whole"))
    if r1 and r2:
        f = lambda r, ag, l: ("%.4f" % r["curve"][ag][l]) if (r["curve"][ag].get(l) is not None and np.isfinite(r["curve"][ag][l])) else "–"
        for l in r1["lams"]:
            lam_rows.append([f"{l:g}"] + [f(r1, ag, l) for ag in ("qrc", "et", "etcnt")] + [f(r2, ag, l) for ag in ("qrc", "et", "etcnt")])
    T["r42_lambda"] = md(["λ", "One-hot: QRC", "ET const.", "ET 1/visits", "Shared features: QRC", "ET const.", "ET 1/visits"], lam_rows)
    def growth(r, ag):                                       # error at the longest trace relative to the agent's best error
        c = [x for x in r["curve"][ag].values() if x is not None and np.isfinite(x)] if r else []
        last = r["curve"][ag].get(max(r["lams"])) if r else None
        return f"{last / min(c):.1f}×" if (c and last is not None and np.isfinite(last)) else "–"
    S["grow"] = {(k, ag): growth(r, ag) for k, r in (("oh", r1), ("ft", r2)) for ag in ("qrc", "et")}
    rng = lambda v: (f"{min(v):+.0f}% to {max(v):+.0f}%" if v else "–")
    S["oh_n"] = len([1 for a, b in onehot_whole if b])
    S["oh_etn_better"] = sum(1 for a, b in onehot_whole if b and b["sig"] and b["rel"] < 0)
    S["oh_etn_range"] = rng([b["rel"] for a, b in onehot_whole if b])
    S["oh_et_better"] = sum(1 for a, b in onehot_whole if a and a["sig"] and a["rel"] < 0)
    S["oh_et_range"] = rng([a["rel"] for a, b in onehot_whole if a])
    S["feat_et"] = " / ".join(pct(a) for a, b in feat_whole)
    S["feat_etn"] = " / ".join(pct(b) for a, b in feat_whole)
    S["feat_same9"] = " / ".join(pct(R42.get((("MultiChainPaperFeat", "", m), "whole"), {}).get("same9_et")) for m in CHAINS)

    # ---------- R43 (pooled with R36): every agent at the same lambda
    sc, dv = defaultdict(dict), defaultdict(int)
    for pre in ("R36", "R43"):
        for (g, cfg), d in minatar.get(pre, {}).items():
            m = re.match(r"(qrc|et|zero|alg3e0|alg3)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)$", cfg)
            if not m or (m.group(1) != "qrc" and m.group(3) != "100"):
                continue
            key = (g, m.group(1), float(m.group(2)), m.group(4))
            sc[key] = score(d); dv[key] = (sum(v["diverged"] for v in d.values()), len(d))

    def best(g, arm, lam=None):
        cand = [(float(np.mean(list(d.values()))), lr, l) for (gg, a, l, lr), d in sc.items() if gg == g and a == arm and (lam is None or l == lam)]
        return max(cand) if cand else None

    arms = [("qrc", "QRC"), ("et", "QRC-ET (split)"), ("alg3", "Algorithm 3 form"), ("alg3e0", "Algorithm 3, η=0"), ("zero", "θ off")]
    lams = sorted({k[2] for k in sc})
    L = dict(lams=lams, rel=defaultdict(dict), div=defaultdict(dict), diff=defaultdict(dict), ndiv=defaultdict(dict))
    for arm, _ in arms:
        for l in lams:
            rel, dif, nd, nr = [], [], 0, 0
            for g in GAMES:
                ref, b, q = best(g, "qrc"), best(g, arm, l), best(g, "qrc", l)
                if ref and b:
                    rel.append(b[0] / ref[0])
                    if q:
                        dif.append((b[0] - q[0]) / ref[0])
                for (gg, a, ll, lr), (x, n) in dv.items():
                    if gg == g and a == arm and ll == l:
                        nd += x; nr += n
            if len(rel) == len(GAMES):
                L["rel"][arm][l] = float(np.mean(rel)); L["diff"][arm][l] = 100 * float(np.mean(dif)) if len(dif) == len(GAMES) else None
            if nr:
                L["div"][arm][l] = 100.0 * nd / nr; L["ndiv"][arm][l] = f"{nd}/{nr}"
    el = [l for l in lams if l > 0]
    T["r43_lambda"] = md(["λ"] + [f"{n} − QRC" for a, n in arms[1:]] + ["Diverged: QRC", "QRC-ET", "Alg. 3"],
                         [[f"{l:g}"] + [(f"{L['diff'][a][l]:+.1f}%" if L["diff"][a].get(l) is not None else "–") for a, _ in arms[1:]]
                          + [L["ndiv"][a].get(l, "–") for a in ("qrc", "et", "alg3")] for l in el])
    grow = []
    for g in GAMES:
        cells = []
        for a, _ in arms:
            b = best(g, a)
            cells.append(f"{b[0]:.1f} (λ={b[2]:g})" if b else "–")
        grow.append([g] + cells)
    T["r43_games"] = md(["Game"] + [n for _, n in arms], grow)
    S["r43"] = {l: L["diff"]["et"].get(l) for l in el}

    # ---------- R40 (feature sharing)
    CODE, crow = {}, []
    for p in ("0.2", "0.9"):
        for D, *_ in CODES:
            fam = defaultdict(list)
            for (pp, DD, ag, lr), d in codes.items():
                if pp == p and DD == D and len(d) >= 2:
                    mu = float(np.mean(list(d.values())))
                    fam["oracle" if ag.startswith("et_oracle") else ("et" if ag.startswith("et") else "qrc")].append((mu, lr, d))
                    if ag == "qrc_lam09":
                        fam["q9"].append((mu, lr, d))
            b = {k: min(v, key=lambda t: t[0]) for k, v in fam.items()}
            if "qrc" in b:
                CODE[(p, D)] = dict(et=paired(b["et"][2], b["qrc"][2]) if "et" in b else None,
                                    oracle=paired(b["oracle"][2], b["qrc"][2]) if "oracle" in b else None,
                                    same9=paired(b["et"][2], b["q9"][2]) if ("et" in b and "q9" in b) else None)
    for D, lab, ov, te in CODES:
        g = lambda p, k: pct(CODE.get((p, D), {}).get(k))
        crow.append([lab, f"{ov:.3f}", f"{te:.0f}%", g("0.2", "et"), g("0.9", "et"), g("0.2", "oracle"), g("0.9", "oracle"), g("0.2", "same9"), g("0.9", "same9")])
    shared = [CODE[(p, D)][k] for p in ("0.2", "0.9") for D, *_ in CODES[1:] for k in ("et",) if CODE.get((p, D), {}).get(k)]
    orac = [CODE[(p, D)]["oracle"] for p in ("0.2", "0.9") for D, *_ in CODES[1:] if CODE.get((p, D), {}).get("oracle")]
    same = [CODE[(p, D)]["same9"] for p in ("0.2", "0.9") for D, *_ in CODES if CODE.get((p, D), {}).get("same9")]
    S["r40_shared"] = rng([c["rel"] for c in shared]); S["r40_oracle"] = rng([c["rel"] for c in orac])
    S["r40_same"] = f"{sum(1 for c in same if c['rel'] < 0)} of {len(same)}"
    T["r40"] = md(["Code", "Overlap", "Best linear θ error", "θ on features: p=0.2", "p=0.9", "θ on state id: p=0.2", "p=0.9",
                   "Both at λ=0.9: p=0.2", "p=0.9"], crow)

    # ---------- MinAtar finals (10 fresh seeds each) and R41 tuning
    FOREST, frow = {}, []
    for pre, name in (("R37", "Conv, QRC (R37)"), ("R41F", "Conv, online Q(λ) (R41)"), ("R38mlpF", "MLP, QRC (R38)"), ("R38lin2F", "Linear, QRC (R38)")):
        R, FOREST[pre], cells = minatar.get(pre, {}), {}, []
        for g in GAMES:
            q = next((d for (gg, c), d in R.items() if gg == g and c.startswith("qrc_best")), None)
            e = next((d for (gg, c), d in R.items() if gg == g and c.startswith("et_best")), None)
            c = paired(score(e), score(q)) if (q and e) else None
            FOREST[pre][g] = c
            cells.append("–" if c is None else (f"**{c['diff']:+.1f}**" if c["sig"] else f"{c['diff']:+.1f}"))
        frow.append([name] + cells)
    T["minatar_final"] = md(["Final comparison (ET − baseline)"] + GAMES, frow)
    r41f, rows41 = minatar.get("R41F", {}), []
    for g in GAMES:
        q = next((d for (gg, c), d in r41f.items() if gg == g and c.startswith("qrc_best")), None)
        e = next((d for (gg, c), d in r41f.items() if gg == g and c.startswith("et_best")), None)
        if q and e:
            c = paired(score(e), score(q))
            rows41.append([g, f"{mean(q):.1f}", f"{mean(e):.1f}", absdiff(c), f"{c['higher']}/{c['n']}" if c else "–"])
    T["r41_final"] = md(["Game", "Best Q(λ)", "Best Q(λ)+ET", "ET − Q(λ) (95% CI)", "ET higher on"], rows41)
    trow, div41 = [], defaultdict(lambda: [0, 0])
    for g in GAMES:
        b = {}
        for (gg, cfg), d in minatar.get("R41", {}).items():
            m = re.match(r"(qrc|et)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)$", cfg)
            if gg != g or not m:
                continue
            if m.group(1) not in b or mean(d) > b[m.group(1)][0]:
                b[m.group(1)] = (mean(d), f"λ={m.group(2)}, lr {m.group(4)}" + (f", ratio {m.group(3)}" if m.group(3) else ""))
            for v in d.values():
                div41[(m.group(1), m.group(2))][0] += v["diverged"]; div41[(m.group(1), m.group(2))][1] += 1
        if "qrc" in b and "et" in b:
            trow.append([g, f"{b['qrc'][0]:.1f} ({b['qrc'][1]})", f"{b['et'][0]:.1f} ({b['et'][1]})"])
    T["r41_tune"] = md(["Game", "Best Q(λ) (setting)", "Best Q(λ)+ET (setting)"], trow)
    l41 = sorted({k[1] for k in div41}, key=float)
    T["r41_div"] = md(["Diverged runs"] + [f"λ={l}" for l in l41],
                      [[n] + [(f"{div41[(a, l)][0]}/{div41[(a, l)][1]}" if div41[(a, l)][1] else "–") for l in l41] for a, n in (("qrc", "Q(λ)"), ("et", "Q(λ)+ET"))])
    return T, S, L, CODE, FOREST


# ------------------------------------------------------------------------------------------------ report blocks
def texts(T, S):
    """All section texts as (kind, content): 'md' = markdown, 'fig' = figure key. Used for the report and RESULTS_R40_R43.md."""
    r43 = S["r43"]
    v = lambda l: (f"{r43[l]:+.1f}%" if r43.get(l) is not None else "–")
    overview = md(["Setting", "QRC-ET against a tuned baseline"], [
        ["One-hot multi-chain, full sweep, 9 tasks (R42)", f"**Better** with a 1/visits θ step on {S['oh_etn_better']} of {S['oh_n']} tasks ({S['oh_etn_range']}); "
                                                             f"with the constant θ step on {S['oh_et_better']} of {S['oh_n']} ({S['oh_et_range']})"],
        ["The paper's shared-feature multi-chain (R42)", f"No advantage: {S['feat_et']} for 8 / 32 / 128 chains"],
        ["Random shared codes (R40)", "No advantage; the gain fades as soon as states share features"],
        ["MinAtar conv, every λ up to 0.95, two forms of the algorithm (R43)", f"No λ where it helps; at λ=0.9 it is worse ({v(0.9)})"],
        ["MinAtar conv, equal tuning, 10 fresh seeds (R37); online Q(λ) (R41)", "Level on all 5 games"],
        ["MinAtar MLP / linear (R38); frozen pre-trained features (R39b)", "Level or worse"]])
    new = [
        ("md", f"# Latest | {TITLE_NEW}"),
        ("md", "**Summary.** An audit against the expected-traces paper found three gaps in the earlier rounds: on the multi-chain QRC had only "
               "λ = 0 and 0.9; on MinAtar QRC-ET had only λ = 0.24 and 0.8 (the paper uses 0.95); and the paper's own multi-chain tasks, with "
               "its own step size for θ, had not been run. R42 and R43 close them. **Expected traces help in the tabular (one-hot) case and "
               "nowhere else we tested**: not with the paper's own shared features, and not on MinAtar at any λ.\n\n" + overview),
        ("md", "## Multi-Chain: Full Sweep With Held-Out Seeds (R42)\n\n"
               "Linear agent, value prediction, 120k steps. Twelve tasks: our sparse-reward task and the paper's task (every reward +1, the last "
               "one +1 or −1), both one-hot, and the paper's shared-feature task (branch + progress + bias + two flags, chain length 16), each "
               "with 8, 32 and 128 chains. Both agents get the same grids: λ ∈ {0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95} and nine learning rates "
               "(factor 2). λ and the learning rate are chosen on seeds 101-110 and the value error is reported on seeds 111-120. "
               "QRC-ET (split form, η=0) runs with the constant θ step used for deep networks (0.1 per visit) and with the 1/visits step of the "
               "paper's tabular experiments (θ = running mean of the traces). 45,360 runs.\n\n"
               "QRC-ET against the best QRC (negative = QRC-ET has the lower value error; bold = 95% interval excludes 0). "
               "Early = after about 1,024 episodes (the paper's measure).\n\n" + T["r42_main"]),
        ("fig", "r42_advantage"),
        ("md", f"**One-hot: the paper's tabular result is replicated.** With the 1/visits step QRC-ET has the lower whole-run error on "
               f"{S['oh_etn_better']} of {S['oh_n']} tasks ({S['oh_etn_range']}). **The θ step matters:** with the constant step QRC-ET falls "
               "behind when states are visited rarely (128 chains), most clearly early in learning. **Shared features: no advantage** over a "
               f"tuned QRC ({S['feat_et']} with the constant step, {S['feat_etn']} with 1/visits), although QRC-ET is still better than QRC at "
               f"the same long trace (λ=0.9: {S['feat_same9']})."),
        ("md", "### Value error against λ (32 chains, whole run, best learning rate at each λ)\n\n" + T["r42_lambda"]),
        ("fig", "r42_lambda"),
        ("md", f"QRC is best with a short trace and gets worse as the trace gets longer: its error at λ=0.95 is {S['grow'][('oh', 'qrc')]} its "
               f"best value with one-hot features and {S['grow'][('ft', 'qrc')]} with shared features. QRC-ET depends much less on λ "
               f"({S['grow'][('oh', 'et')]} and {S['grow'][('ft', 'et')]}). So QRC-ET repairs the long traces in both cases, but with shared "
               "features QRC with a short trace is just as good, and nothing is gained once λ is tuned."),
        ("md", "### θ step size (QRC-ET at λ=0.9 against the best QRC, early window)\n\n" + T["r42_step"] +
               "\nNo constant step is good everywhere. With 128 chains a state is visited only about 8 times in this window: the largest "
               "step is then the best of the three, and the small ones are far behind. With 8 chains the smaller steps are as good or "
               "better. The 1/visits step adapts by itself."),
        ("md", "## MinAtar: The Missing λ Values, a θ-Off Control and the Paper's Algorithm 3 Form (R43)\n\n"
               "Same games, learning rates and seeds (301-303) as the R36 tuning, so the R36 QRC runs are the baseline. All QRC-ET arms: "
               "θ/Q learning-rate ratio 100, η=0.3 unless stated. Split form at λ = 0.5, 0.9, 0.95 (0.24 and 0.8 are in R36); the same agent "
               "with θ's output set to zero; the paper's Algorithm 3 form (one-step lag, θ trained on the mixed trace), also with η=0. "
               "900 runs, 5M steps. Differences at the same λ, as % of each game's best QRC score, mean of 5 games; "
               "diverged runs of 60 (Q value not finite or above 1e4).\n\n" + T["r43_lambda"]),
        ("fig", "r43_lambda"),
        ("md", f"**There is no λ where expected traces help.** Up to λ=0.8 the differences are small and change sign from game to game "
               f"({v(0.24)}, {v(0.5)}, {v(0.8)}); at λ=0.9 QRC-ET is worse ({v(0.9)}) and at λ=0.95 both agents collapse ({v(0.95)}). QRC-ET "
               "does not make long traces usable: it diverges more often than QRC there. The paper's Algorithm 3 form behaves like the split "
               "form. **θ off** collapses on most games from λ=0.5 on, as in R28: the head then has a short trace while the rest of the "
               "network keeps the long one. So θ's output is really used as part of the trace, but this arm cannot serve as a control; the "
               "fair control is QRC with a short trace (λ=0.24), which is as good as QRC-ET at λ=0.8.\n\n"
               "Best setting of every agent over all λ and learning rates (3 tuning seeds; a screen, not a test):\n\n" + T["r43_games"]),
        ("md", "## Reproduction | R42-R43\n\n" + md(["Experiment", "Job script", "Runs"], [
            ["R42 multi-chain sweep", "slurm/R42_multichain_lambda.sh", "45,360"],
            ["R43 MinAtar λ / θ off / Algorithm 3", "slurm/R43_minatar_lambda.sh", "900"]]) +
               "\nNew: `--et_count_step` (apply_count_step_patch.py), the paper's tasks in `multichain_env.py` (MultiChainPaper, "
               "MultiChainPaperFeat). Summaries: `python summarize_r42.py`, `python summarize_lambda.py R36 R43`. "
               "The 45,360 short R42 runs are not uploaded one by one; `r42_full.txt` in the repository has every per-λ table. "
               f"Code: branch [cluster-r29]({GITHUB}), `RESULTS_R40_R43.md`."),
    ]
    why = [
        ("md", f"# Follow-up | {TITLE_WHY}"),
        ("md", "**Summary.** Two experiments suggested by Peter to find out why the one-hot gain does not carry over. (1) Shared features: "
               "give every state a random multi-hot code and make the codes shorter. (2) Remove QRC's gradient correction: online Q(λ) on "
               "MinAtar. **The gain disappears as soon as states share features, and Q(λ) behaves like QRC.**"),
        ("md", "## Feature Sharing on the Multi-Chain (R40, R40b, R40c)\n\n"
               "Sparse multi-chain, 32 chains, linear agent. Each state gets a fixed random code of length D with k ones (one-hot: D=130, k=1; "
               "then D=64, 32, 16 with k=3 and D=10 with k=5). QRC λ ∈ {0, 0.9}; QRC-ET split form, λ=0.9, η=0, three θ steps; learning rates "
               "0.0003 to 0.1; 10 seeds (settings chosen and reported on the same seeds). R40c: θ reads a one-hot state id while Q keeps the "
               "shared code. 2,920 runs. QRC-ET against the best QRC, and both agents at λ=0.9:\n\n" + T["r40"]),
        ("fig", "r40_sharing"),
        ("md", f"With one-hot codes QRC-ET is clearly better. With any sharing the advantage is gone: against the best QRC the difference "
               f"ranges from {S['r40_shared']}. Giving θ a one-hot state id (so that it can represent the expected trace exactly) does not bring the "
               f"one-hot gain back ({S['r40_oracle']}), so θ's representation is not the main reason. At the same λ=0.9 QRC-ET is better "
               f"than QRC in {S['r40_same']} cases, and that gap shrinks as sharing grows: shared features already spread an update over "
               "related states, so a long trace is no longer needed."),
        ("md", "## Online Q(λ) on MinAtar (R41)\n\n"
               "Watkins Q(λ) without the gradient correction (`--q_lambda`), with and without expected traces; the same equal-budget design "
               "as R36 (24 settings each, 3 seeds), then each method's best setting on 10 fresh seeds (401-410).\n\n" + T["r41_final"]),
        ("fig", "minatar_forest"),
        ("md", "Every final MinAtar comparison so far (10 fresh seeds each; ET − baseline score, bold = significant; in the figure each "
               "panel has its own scale):\n\n" + T["minatar_final"]),
        ("md", "### Q(λ) tuning (3 seeds)\n\n" + T["r41_tune"] + "\n" + T["r41_div"] +
               "\nWithout the gradient correction long traces diverge even more often, with or without expected traces."),
        ("md", "## Reproduction | R40-R41\n\n" + md(["Experiment", "Job script", "Runs"], [
            ["R40 / R40b multi-hot codes", "slurm/R40_multihot.sh, R40b_multihot_tuning.sh", "1,200 / 1,120"],
            ["R40c θ on a one-hot state id", "slurm/R40c_oracle_theta.sh", "600"],
            ["R41 online Q(λ): tuning / final", "slurm/R41_qlambda_tune.sh, R41_qlambda_final.sh", "720 / 100"]]) +
               "\nNew agent options: `--q_lambda` (apply_qlambda_patch.py), `--q_channels`, `--et_theta_input` (apply_oracle_theta_patch.py). "
               "Summaries: `python summarize_r40.py R40 R40b R40c`, `python summarize_r36.py R41`, `python summarize_r34c.py R41F`."),
    ]
    return new, why


def write_results(new, why, path="RESULTS_R40_R43.md"):
    out = ["# Expected traces with QRC: R40-R43\n", "Generated by `wb_add_r40_r43.py` from the cluster logs (same tables as the W&B report). "
           "γ = 0.99 everywhere. Negative % = QRC-ET has the lower value error; bold = 95% interval excludes 0.\n"]
    for kind, c in new + why:
        if kind == "md":
            if c.startswith("# Latest | ") or c.startswith("# Follow-up | "):
                c = "## " + c.split(" | ", 1)[1]
            elif c.startswith("### "):
                c = "#### " + c[4:]
            elif c.startswith("## "):
                c = "### " + c[3:]
            out.append(c + "\n")
        else:
            out.append(f"*(figure: report_figures/{c}.png in the W&B report)*\n")
    open(path, "w").write("\n".join(out))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=REPORT_URL)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    args.url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]

    minatar, codes = read_runs()
    R42, n42 = analyse_r42()
    print("complete runs: R42 %d, " % n42 + ", ".join(f"{p} {sum(len(d) for d in minatar.get(p, {}).values())}"
          for p in ("R36", "R43", "R41", "R41F", "R37", "R38mlpF", "R38lin2F")) + f", R40+R40b+R40c {sum(len(d) for d in codes.values())}")
    T, S, L, CODE, FOREST = build(minatar, codes, R42)
    F = make_figures(R42, L, CODE, FOREST)
    for k, t in T.items():
        print(f"\n### {k}\n{t}")
    print("figures:", ", ".join(F.values()))
    new, why = texts(T, S)
    print("wrote", write_results(new, why))

    import wandb_workspaces.reports.v2 as wr
    rs = wr.Runset(entity=args.entity, project=args.project, name="Figures | R40-R43", filters=f"Config('experiment') == '{TAG}'")

    def to_blocks(items):
        out = [wr.HorizontalRule()]
        for kind, c in items:
            if kind == "md":
                out.append(wr.MarkdownBlock(text=c))
            elif c in F:
                out.append(wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title="", media_keys=[f"fig/{c}"], num_columns=1,
                                        layout=wr.Layout(x=0, y=0, w=24, h=19 if c == "r42_advantage" else 11))]))
        return out
    blocks = to_blocks(new) + to_blocks(why)
    if args.dry_run:
        print(f"\n{len(blocks)} blocks built (nothing changed online).")
        return

    rep = wr.Report.from_url(args.url)
    bl = list(rep.blocks)
    txt = [block_text(b) for b in bl]
    if any(TITLE_NEW in t or TITLE_WHY in t for t in txt):
        raise SystemExit("The R40-R43 sections are already in the report; nothing changed.")
    i = next((k for k, t in enumerate(txt) if NEXT_SECTION in t), None)
    if i is None:
        raise SystemExit("Could not find the R38-R39 section heading; nothing changed.")
    bl[i].text = bl[i].text.replace("# Latest | ", "# Follow-up | ")
    if i > 0 and type(bl[i - 1]).__name__ == "HorizontalRule":
        i -= 1
    if not args.no_log:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, name="FOLLOW-UP | R40-R43 report figures",
                         job_type="report-data", group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
        run.log({f"fig/{k}": wandb.Image(p) for k, p in F.items()})
        run.finish()
    backup = wr.Report.from_url(args.url)
    backup.title = backup.title + " (backup before R40-R43)"
    backup.save(clone=True)
    print("Backup of the current report:", backup.url)
    rep.blocks = bl[:i] + blocks + bl[i:]
    rep.width = "fluid"
    rep.save()
    print(f"Inserted {len(blocks)} blocks at position {i}. Updated report:", rep.url)


if __name__ == "__main__":
    main()
