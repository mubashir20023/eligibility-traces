"""Rebuild the follow-up part of the existing report so it reads well on screen and in the PDF export.

  * Order: the original five-game overview stays first; then the NEWEST follow-up (R33-R35), then R28-R31 (once),
    then the original per-game comparisons and the rest of the original report.
  * Tables are compact Markdown tables that fit the page width (no scrolling, no clipped columns).
  * Plots are figure pairs (left/right) with distinct colours and legends, logged as images.
  * Removes every earlier appended follow-up block (including the duplicated R28-R31 copy).

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_reorganize.py --inspect     # list the report's blocks, change nothing
    ~/wbenv/bin/python wb_reorganize.py --dry-run     # build tables/figures locally, change nothing
    ~/wbenv/bin/python wb_reorganize.py               # log figures run, back up the report, rebuild, save
"""
import argparse, os

import numpy as np

import wb_append_report as A           # R28-R31 data
import wb_append_r33_r35 as B          # R33-R35 data

REPORT_URL = A.REPORT_URL
TAG = "FOLLOWUP-R28-R35-FIGS"
GITHUB = A.GITHUB
FIGDIR = "report_figures"
INSERT_BEFORE = "Matched Game Comparisons"     # original section the follow-ups go in front of
FIRST_APPENDED = "Follow-up | Freeway Controls"  # first heading of the earlier appended content

COLORS = [("split", "#d62728"), ("best qrc-et", "#d62728"), ("qrc-et (", "#d62728"), ("ordinary-trace", "#d62728"),
          ("no lag", "#7f7f7f"), ("current code", "#ff7f0e"), ("θ off", "#7f7f7f"), ("e_w", "#ff7f0e"),
          ("λ=0.9", "#9467bd"), ("λ=0.8", "#9467bd"), ("λ=0.24", "#2ca02c"), ("λ=0)", "#1f77b4"),
          ("best qrc", "#1f77b4"), ("qrc", "#1f77b4")]
SEED_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def color(label, i):
    l = label.lower()
    if l.startswith("seed"):
        return SEED_COLORS[i % len(SEED_COLORS)]
    for k, c in COLORS:
        if k in l:
            return c
    return SEED_COLORS[i % len(SEED_COLORS)]


# ------------------------------------------------------------------------------------------------ figures
def fig_pair(path, panels):
    """panels: list of (title, x, {label: y}, xlabel, ylabel, opts). Saves a 1 x len(panels) figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (title, x, lines, xl, yl, o) in zip(axes, panels):
        for i, (lab, y) in enumerate(lines.items()):
            if y is None:
                continue
            y = np.array([np.nan if v is None else v for v in y], float)
            dashed = "no lag" in lab.lower() or "θ off" in lab.lower()
            ax.plot(x[:len(y)], y, color=color(lab, i), lw=1.6, ls="--" if dashed else "-",
                    marker="o" if o.get("markers") else None, ms=4, label=lab)
        if o.get("logx"):
            ax.set_xscale("log")
        if o.get("logy"):
            ax.set_yscale("log")
        if o.get("ylim"):
            ax.set_ylim(*o["ylim"])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(alpha=.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def S(series, key):
    return series.get(key) or {"x": []}


def make_figures(D1, D2):
    os.makedirs(FIGDIR, exist_ok=True)
    s2, s1 = D2["series"], D1["series"]
    ve = ("Environment steps", "value error over all states")
    ret = ("Environment steps", "rolling-100-episode return")
    F = {}

    def add(key, panels):
        if any(any(v is not None for v in p[2].values()) for p in panels):
            F[key] = fig_pair(os.path.join(FIGDIR, key + ".png"), panels)

    # ---- R33-R35
    a, b = S(s2, "oh8"), S(s2, "oh32")
    lab = {"qrc": "best QRC", "split": "QRC-ET, split (both fixes)", "lag": "QRC-ET, current code"}
    add("onehot_curves", [(f"One-hot, {t}: value error (20 seeds)", np.array(s["x"]), {lab[k]: s.get(k) for k in lab}, *ve, {"ylim": (0, 0.12)})
                          for t, s in [("m=8, p=0.2", a), ("m=32, p=0.9", b)]])
    c = S(s2, "conv32")
    labc = {"qrc0": "QRC (λ=0)", "qrc09": "QRC (λ=0.9)", "nolag": "QRC-ET, no lag", "lag": "QRC-ET, current code"}
    add("conv_curves", [("Conv network, m=32, p=0.9: value error (10 seeds)", np.array(c["x"]), {labc[k]: c.get(k) for k in labc}, *ve, {"ylim": (0, 0.12)})])
    labm = {"qrc": "best QRC", "split03": "QRC-ET, split η=0.3", "nolag03": "QRC-ET, no lag η=0.3"}
    g = {k: S(s2, "mn_" + k) for k in ("freeway", "breakout", "asterix")}
    add("minatar_fw_ast", [(f"{n.capitalize()} (5 seeds)", np.array(g[n]["x"]), {labm[k]: g[n].get(k) for k in labm}, *ret, {})
                           for n in ("freeway", "asterix")])
    e0 = S(s2, "fw_eta0")
    add("minatar_br_eta0", [("Breakout (5 seeds)", np.array(g["breakout"]["x"]), {labm[k]: g["breakout"].get(k) for k in labm}, *ret, {}),
                            ("Freeway, split η=0, lr 2e-3: each seed", np.array(e0["x"]),
                             {f"seed {k[1:]}": v for k, v in e0.items() if k != "x"}, *ret, {})])
    # ---- R28-R31
    fw = S(s1, "fw")
    labf = {"et": "QRC-ET (η=0.3, lr 0.002)", "qrc08": "QRC (λ=0.8, lr 0.0005)", "qrc024": "QRC (λ=0.24, lr 0.002)", "etoff": "QRC-ET, Θ off"}
    add("r28_freeway", [("Freeway: QRC-ET vs QRC (15 seeds)", np.array(fw["x"]), {labf[k]: fw.get(k) for k in ("et", "qrc08", "qrc024")}, *ret, {}),
                        ("Freeway: QRC-ET vs QRC-ET with Θ off", np.array(fw["x"]), {labf[k]: fw.get(k) for k in ("et", "etoff")}, *ret, {})])
    lr = S(s1, "lr")
    labl = {"et": "QRC-ET (η=0.3)", "qrc08": "QRC (λ=0.8)", "qrc024": "QRC (λ=0.24)"}
    add("r29_lr", [("Freeway: final score vs learning rate", np.array(lr["x"], float), {labl[k]: lr.get(k) for k in labl},
                    "Q learning rate", "final score", {"logx": True, "markers": True})])
    labmc = {"best_qrc": "best QRC", "qrc08": "QRC (λ=0.8)", "best_et": "best QRC-ET"}
    add("r30_chains", [(f"Value error vs number of chains, σ={sg}", np.array(S(s1, k)["x"], float), {labmc[j]: S(s1, k).get(j) for j in labmc},
                        "number of chains", "value error (AUC)", {"logx": True, "markers": True}) for sg, k in [(1, "m_s1"), (0, "m_s0")]])
    add("r30_curves", [(f"m={m}, σ=1: value error", np.array(S(s1, f"mc{m}")["x"]), {labmc[j]: S(s1, f"mc{m}").get(j) for j in labmc},
                        *ve, {"ylim": (0, 0.3)}) for m in (32, 128)])
    tl = S(s1, "tl")
    add("r31_trace_loss", [("m=32, σ=1: Θ trace loss (Θ lr 0.1)", np.array(tl["x"]),
                            {"ordinary-trace target": tl.get("raw"), "Algorithm 3 e_w target": tl.get("e_w")},
                            "Environment steps", "trace loss", {"logy": True})])
    return F


# ------------------------------------------------------------------------------------------------ tables
def md(cols, rows):
    return ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" +
            "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows))


def pct(a, b):
    try:
        return f"{100 * (float(a) / float(b) - 1):+.1f}%"
    except (ValueError, ZeroDivisionError):
        return "–"


def tables_r33_r35(D2):
    T = D2["tables"]
    out = {}
    cols, rows = T["One-Hot Multi-Chain | Which Fix Matters (R33)"]
    out["r33"] = md(["Task", "Best QRC (error, setting)", "Current code (lag)", "Lag, Θ step fixed", "No lag, Θ step ×3", "**No lag, Θ step fixed**"],
                    [[r[0], f"{r[1]} ({r[2]})", r[3], r[4], r[5], f"**{r[6]}**"] for r in rows])
    cols, rows = T["One-Hot Multi-Chain | 20 Fresh Seeds (R34a, R35a)"]
    out["fresh"] = md(["Task", "Best QRC (error, setting)", "Current code", "No lag", "**Split (both fixes)**"],
                      [[r[0], f"{r[1]} ({r[2]})", r[8], r[4], f"**{r[6]}**"] for r in rows])
    cols, rows = T["Conv Network | Multi-Chain (R34b)"]
    out["conv"] = md(["Task", "Best QRC (error, setting)", "QRC λ=0.9", "QRC-ET current code", "QRC-ET no lag, Θ lr 0.1", "QRC-ET no lag, Θ lr 0.3"],
                     [[r[0], f"{r[1]} ({r[2]})", pct(r[3], r[1]), r[4], r[5], r[6]] for r in rows])
    cols, rows = T["MinAtar | Final Score with the Fixes (R34c, R35)"]
    games = [g for g in B.GAMES if any(r[0] == g for r in rows)]
    cell = {}
    for g, ag, lr, sc, dv, ci in rows:
        d = dv.split("/")[0]
        cell[(ag, g)] = f"{sc}" + (f" ({dv} diverged)" if d not in ("0", "") else "") + (f" · Δ {ci.split(' [')[0]}" if ci != "–" else "")
    agents = [B.nice(a) for a in ["qrc_lam024", "qrc_lam08", "et_split_eta03", "et_split_eta0", "et_lag_eta03", "et_fix_eta03", "et_fix_eta0"]]
    out["minatar"] = md(["Agent (best learning rate per game)"] + games,
                        [[f"**{a}**" if "split η=0.3" in a else a] + [cell.get((a, g), "–") for g in games] for a in agents if any((a, g) in cell for g in games)])
    out["repro"] = md(*T["Reproduction | R33-R35"])
    return out


def tables_r28_r31(D1):
    T = D1["tables"]
    out = {"fw": md(*T["Freeway | 15 Fresh Seeds"]), "lr": md(*T["Freeway | Score by Learning Rate"])}
    cols, rows = T["Multi-Chain | Value Error (lower is better)"]
    out["mc"] = md(["Chains", "Noise σ", "Best QRC (error, setting)", "Best QRC-ET (error, setting)", "ET vs QRC", "ET better on"],
                   [[r[0], r[1], f"{r[2]} ({r[3]})", f"{r[4]} ({r[5]})", r[6], r[7]] for r in rows])
    out["theta"] = md(*T["Θ Target and Learning Rate | m=32, σ=1"])
    out["repro"] = md(*T["Reproduction | R28-R31"])
    return out


# ------------------------------------------------------------------------------------------------ blocks
def build_blocks(wr, F, t2, t1, entity, project):
    rs = wr.Runset(entity=entity, project=project, name="Figures | R28-R35", filters=f"Config('experiment') == '{TAG}'")
    M = lambda text: wr.MarkdownBlock(text=text)

    def fig(key, h=11):
        if key not in F:
            return None
        return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title="", media_keys=[f"fig/{key}"], num_columns=1,
                                                                  layout=wr.Layout(x=0, y=0, w=24, h=h))])

    new = [
        wr.HorizontalRule(),
        M("# Latest | Expected Traces with the Fixes (R33-R35)"),
        M("**Summary.** With features that keep states apart (one-hot), compiled expected traces beat tuned QRC by "
          "**13-32%** on every task and **every one of 20 fresh seeds**. This needed two fixes: remove Algorithm 3's "
          "one-step lag, and give Θ a correctly sized step. A third change, the **split form** (Θ predicts only the past "
          "part of the trace; the current gradient stays exact), makes it stable with deep networks. On MinAtar with the "
          "conv network, the fixed QRC-ET is stable and level with tuned QRC, but does not beat it."),
        M("## One-Hot Multi-Chain | Which Fix Matters (R33, 10 seeds)"),
        M("Peter's suggestion: no conv layers, so each state is a one-hot input (tabular). Mathieu's suggestion: sparse "
          "reward (+1 with probability p at the shared state). Cells: QRC-ET's value error relative to the best tuned QRC "
          "(any λ, any learning rate); negative = better.\n\n" + t2["r33"]),
        M("Both fixes are needed: fixing only Θ's step size still loses by 44-80%."),
        M("## One-Hot | 20 Fresh Seeds and the Split Form (R34a, R35a)"),
        M("Seeds 11-30, each agent at the learning rate R33 selected. Cells: relative value error (seeds where QRC-ET is "
          "better).\n\n" + t2["fresh"]),
        fig("onehot_curves"),
        M("## Conv Network on the Same Task (R34b, 10 seeds)"),
        M("Same sparse tasks with the usual MinAtar conv network. The best QRC uses no trace at all (λ=0); every QRC-ET "
          "version is worse than it but better than QRC with λ=0.9. The conv features collapse (trace diversity → 0, "
          "effective rank 61 → 3-8), so the network already shares credit between chains.\n\n" + t2["conv"]),
        fig("conv_curves"),
        M("## MinAtar with the Fixes (R34c, R35; 5 seeds, 5M steps)"),
        M("Final score (mean ± s.e.), Δ = difference to the best QRC. Removing the lag alone makes QRC-ET diverge on "
          "Freeway and Asterix; the split form never diverged. ET was only run with λ=0.8; at matched λ=0.8 it is level "
          "with QRC on all three games.\n\n" + t2["minatar"]),
        fig("minatar_fw_ast"),
        fig("minatar_br_eta0"),
        M("With η=0 (pure expected trace), 4 of 10 Freeway runs learned and then collapsed to return 0 (Q → 0) rather than "
          "diverging (right). Keeping 30% of the ordinary trace (η=0.3) avoids it at lr 5e-4."),
        M("## Reproduction | R33-R35\n\n" + t2["repro"] +
          f"\nCode: branch [cluster-r29]({GITHUB}) (`RESULTS_R33_R35.md`). All 2,030 runs are in this project, tagged R33 … R35."),
    ]
    old = [
        wr.HorizontalRule(),
        M("# Follow-up | Freeway Controls and Multi-Chain (R28-R31)"),
        M("## Freeway Controls (R28, 15 fresh seeds)"),
        M("Original Algorithm-3 agent at the R27 setting, 5M steps. Does the R27 Freeway gain come from the expected trace?\n\n" + t1["fw"]),
        M("QRC-ET beats the usual QRC baseline (λ=0.8), but QRC with a shorter trace (λ=0.24) matches it within noise. "
          "With Θ switched off every run diverged."),
        fig("r28_freeway"),
        M("## Freeway | Learning-Rate Sweep (R29)\n\n" + t1["lr"]),
        fig("r29_lr"),
        M("## Multi-Chain Prediction (R30)"),
        M("The expected-traces paper's multi-chain task with the same deep agents: m chains merge into one state, terminal "
          "reward 1 + σ·noise. Value error over all states (lower is better), 5 seeds.\n\n" + t1["mc"]),
        fig("r30_chains"),
        fig("r30_curves"),
        M("## Θ Target and Learning Rate (R31, m=32, σ=1)\n\n" + t1["theta"]),
        fig("r31_trace_loss"),
        M("At that point, with the lag still in the agent, QRC-ET did not beat tuned QRC; R33-R35 above identify and fix the cause."),
        M("## Reproduction | R28-R31\n\n" + t1["repro"]),
        wr.HorizontalRule(),
    ]
    return [b for b in new if b is not None], [b for b in old if b is not None]


def block_text(b):
    for attr in ("text", "title"):
        t = getattr(b, attr, None)
        if t:
            return t if isinstance(t, str) else " ".join(str(x) for x in (t if isinstance(t, list) else [t]))
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=REPORT_URL)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="reuse an already logged figures run")
    args = ap.parse_args()
    args.url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]
    import wandb_workspaces.reports.v2 as wr

    if args.inspect:
        for i, b in enumerate(wr.Report.from_url(args.url).blocks):
            print(i, type(b).__name__, block_text(b)[:80].replace("\n", " / "))
        return

    D1 = A.build_data(A.load_runs())
    D2 = B.build_data(B.load_runs())
    F = make_figures(D1, D2)
    t2, t1 = tables_r33_r35(D2), tables_r28_r31(D1)
    for k, v in {**{"R33-35 " + k: v for k, v in t2.items()}, **{"R28-31 " + k: v for k, v in t1.items()}}.items():
        print(f"\n### {k}\n{v}")
    print("figures:", ", ".join(F.values()))
    new, old = build_blocks(wr, F, t2, t1, args.entity, args.project)
    if args.dry_run:
        print(f"\n{len(new)} + {len(old)} blocks built. Figures are in {FIGDIR}/ (open them to check).")
        return

    rep = wr.Report.from_url(args.url)
    blocks = list(rep.blocks)
    texts = [block_text(b) for b in blocks]
    first = next((i for i, t in enumerate(texts) if FIRST_APPENDED in t), None)
    ins = next((i for i, t in enumerate(texts) if INSERT_BEFORE in t), None)
    if first is None or ins is None or ins > first:
        raise SystemExit(f"Could not find the expected sections (appended at {first}, insert point at {ins}); run --inspect.")
    if first > 0 and type(blocks[first - 1]).__name__ == "HorizontalRule":
        first -= 1
    original = blocks[:first]                     # everything before the earlier appended follow-ups
    print(f"keeping {len(original)} original blocks, removing {len(blocks) - first} appended blocks, inserting at {ins}")

    if not args.no_log:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, name="FOLLOW-UP | R28-R35 report figures",
                         job_type="report-data", group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
        run.log({f"fig/{k}": wandb.Image(p) for k, p in F.items()})
        run.finish()

    backup = wr.Report.from_url(args.url)
    backup.title = backup.title + " (backup before reorganising)"
    backup.save(clone=True)
    print("Backup of the current report:", backup.url)
    rep = wr.Report.from_url(args.url)
    rep.blocks = original[:ins] + new + old + original[ins:]
    rep.width = "fluid"
    rep.save()
    print("Updated report:", rep.url)


if __name__ == "__main__":
    main()
