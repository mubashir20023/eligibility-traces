"""Add the R42q (plain TD(lambda) baseline on the multi-chain) + R44 (10-seed MinAtar confirmation) section to the report,
as the newest follow-up: inserted right before the R42-R43 section, same layout. Also writes RESULTS_R42q_R44.md.

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_add_r42q_r44.py --dry-run     # tables + figures + RESULTS_R42q_R44.md; changes nothing online
    ~/wbenv/bin/python wb_add_r42q_r44.py               # log the figures run, back up the report, insert, save
Needs r42_cache.pkl and r42q_cache.pkl (written by: python summarize_r42.py  /  python summarize_r42.py R42q).
"""
import argparse, os, re, sys
from collections import defaultdict

import numpy as np

from wb_reorganize import md, block_text, REPORT_URL, GITHUB
from wb_add_r40_r43 import (analyse_r42, paired, pct, load_minatar, style, finish, FAMILIES, CHAINS, GAMES, TEST,
                            COL, INK, MUTED, FIGDIR)

TAG = "FOLLOWUP-R42q-R44-FIGS"
TITLE = "Plain TD(λ) Baseline and 10-Seed MinAtar Confirmation (R42q, R44)"
NEXT_SECTION = "Full λ Sweeps, the Paper's Own Tasks and the Missing MinAtar Cells"      # R42-R43 heading
R44_TAGS = [("lam09", "both at λ=0.9"), ("lam095", "both at λ=0.95"), ("best", "best setting of each")]
FEAT = ("MultiChainPaperFeat", "")


def absci(c):
    if c is None:
        return "–"
    s = f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]"
    return f"**{s}**" if c["sig"] else s


def read_r44():
    pat = re.compile(r"^(\w+)-v1__R44_(lam09|lam095|best)_(qrc|et)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)__(\d+)$")
    runs = defaultdict(dict)
    for e in os.scandir("logs/MinAtar"):
        m = pat.match(e.name)
        if not m:
            continue
        r = load_minatar(f"logs/MinAtar/{e.name}/training_data.csv")
        if r is None:
            continue
        g, tag, meth, lam, ratio, lr, seed = m.groups()
        runs[(g, tag, meth)][int(seed)] = dict(score=r[0], diverged=r[1],
                                               setting=f"λ={lam}, lr {lr}" + (f", ratio {ratio}" if ratio else ""))
    return runs


def build(RQ, RT, r44):
    T, S, F = {}, {}, {}
    # ---------------- R42q against R42: ET on top of TD(lambda) and on top of QRC (whole run)
    rows = []
    for fam, name in FAMILIES:
        for m in CHAINS:
            c = lambda R, ag: (R.get(((fam[0], fam[1], m), "whole"), {}).get(ag) or {}).get("cmp")
            rows.append([name if m == CHAINS[0] else "", m, pct(c(RT, "et")), pct(c(RT, "etcnt")), pct(c(RQ, "et")), pct(c(RQ, "etcnt"))])
    T["rel"] = md(["Task", "Chains", "TD(λ) + ET const. vs best TD(λ)", "TD(λ) + ET 1/visits", "QRC + ET const. vs best QRC", "QRC + ET 1/visits"], rows)

    # ---------------- shared-feature task: best errors of all six agents, and QRC (no ET) against TD(lambda) + ET
    rows, fig = [], defaultdict(list)
    S["qrc_vs_tdet"] = []
    for m in CHAINS:
        q, t = RQ.get(((FEAT[0], FEAT[1], m), "whole")), RT.get(((FEAT[0], FEAT[1], m), "whole"))
        if not (q and t):
            continue
        g = lambda R, ag: (R.get(ag) or {}).get("mean")
        f4 = lambda v: f"{v:.4f}" if v is not None else "–"
        c1 = paired(q["qrc"]["seeds"], t["et"]["seeds"], TEST) if t.get("et") else None        # QRC alone vs TD(lambda) + ET
        c2 = paired(q["qrc"]["seeds"], t["etcnt"]["seeds"], TEST) if t.get("etcnt") else None
        rows.append([m, f4(g(t, "qrc")), f4(g(t, "et")), f4(g(t, "etcnt")), f4(g(q, "qrc")), f4(g(q, "et")), f4(g(q, "etcnt")), pct(c1), pct(c2)])
        S["qrc_vs_tdet"] += [c for c in (c1, c2) if c]
        for key, R, ag in (("td", t, "qrc"), ("tdet", t, "et"), ("qrc", q, "qrc"), ("qrcet", q, "et")):
            fig[key].append(g(R, ag))
    T["feat"] = md(["Chains", "TD(λ)", "TD(λ)+ET const.", "TD(λ)+ET 1/visits", "QRC", "QRC+ET const.", "QRC+ET 1/visits",
                    "QRC vs TD(λ)+ET const.", "QRC vs TD(λ)+ET 1/visits"], rows)
    feat_td = [(RT.get(((FEAT[0], FEAT[1], m), "whole"), {}).get("et") or {}).get("cmp") for m in CHAINS]
    S["feat_td"] = " / ".join(pct(c) for c in feat_td)
    S["feat_td_seeds"] = ", ".join(f"{c['lower']}/{c['n']}" for c in feat_td if c)
    feat_q = [(RQ.get(((FEAT[0], FEAT[1], m), "whole"), {}).get("et") or {}).get("cmp") for m in CHAINS]
    S["feat_q"] = " / ".join(pct(c) for c in feat_q)
    v = [c["rel"] for c in S["qrc_vs_tdet"]]
    S["qrc_vs_tdet_rng"] = f"{-max(v):.0f}% to {-min(v):.0f}%" if v else "–"
    S["qrc_vs_tdet_sig"] = f"{sum(1 for c in S['qrc_vs_tdet'] if c['sig'] and c['rel'] < 0)} of {len(S['qrc_vs_tdet'])}"
    oh = [abs(RQ[k]["qrc"]["mean"] / RT[k]["qrc"]["mean"] - 1) * 100 for k in RQ
          if k[1] == "whole" and not k[0][0].endswith("Feat") and k in RT and RQ[k]["qrc"]["mean"] and RT[k]["qrc"]["mean"]]
    S["oh_same"] = f"{max(oh):.1f}%" if oh else "–"

    # ---------------- R44
    rows, S["r44"], forest = [], defaultdict(list), defaultdict(dict)
    for tag, name in R44_TAGS:
        for g in GAMES:
            q, e = r44.get((g, tag, "qrc"), {}), r44.get((g, tag, "et"), {})
            c = paired({s: v["score"] for s, v in e.items()}, {s: v["score"] for s, v in q.items()})
            if c is None:
                continue
            mean = lambda d: np.mean([v["score"] for v in d.values()])
            nd = lambda d: sum(v["diverged"] for v in d.values())
            rows.append([name if g == GAMES[0] else "", g, f"{mean(q):.1f} [{nd(q)}]", f"{mean(e):.1f} [{nd(e)}]", absci(c),
                         f"{c['higher']}/{c['n']}", next(iter(q.values()))["setting"], next(iter(e.values()))["setting"]])
            forest[tag][g] = c
            S["r44"][tag].append(c["rel"])
    T["r44"] = md(["Comparison", "Game", "QRC [div.]", "QRC-ET [div.]", "ET − QRC (95% CI)", "ET higher on", "QRC setting", "QRC-ET setting"], rows)
    S["r44_mean"] = {tag: (f"{np.mean(v):+.0f}%" if v else "–") for tag, v in S["r44"].items()}
    S["r44_sig"] = {tag: ", ".join(f"{g} {'better' if c['diff'] > 0 else 'worse'}" for g, c in forest[tag].items() if c["sig"]) or "none"
                    for tag, _ in R44_TAGS}

    # ---------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    if fig:
        figr, ax = plt.subplots(1, 1, figsize=(7.4, 4.3))
        xs = np.arange(len(fig["qrc"]))
        for key, col, ls, mk, lab in (("td", COL["qrc"], "--", "o", "TD(λ)"), ("tdet", COL["et"], "--", "s", "TD(λ) + expected traces"),
                                      ("qrc", COL["qrc"], "-", "o", "QRC"), ("qrcet", COL["et"], "-", "s", "QRC + expected traces")):
            ax.plot(xs, fig[key], color=col, ls=ls, lw=2, marker=mk, ms=7.5, mec="white", mew=1.3, label=lab, zorder=3)
        ax.set_xticks(xs); ax.set_xticklabels([str(m) for m in CHAINS][:len(xs)]); ax.set_xlim(-0.3, len(xs) - 0.7)
        style(ax, "Paper's shared-feature multi-chain: best value error (test seeds)", "number of chains (m)",
              "value error, whole run (lower is better)")
        F["r42q_feat"] = finish(figr, [ax], os.path.join(FIGDIR, "r42q_feat.png"), top=0.86, handlelength=4.5); plt.close(figr)
    if forest:
        figr, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), sharey=True)
        ys = np.arange(len(GAMES))[::-1]
        for ax, (tag, name) in zip(axes, R44_TAGS):
            c = [forest[tag].get(g) for g in GAMES]
            x = np.array([np.nan if v is None else v["rel"] for v in c]); lo = np.array([np.nan if v is None else v["rel_lo"] for v in c])
            hi = np.array([np.nan if v is None else v["rel_hi"] for v in c]); ok = np.isfinite(x)
            ax.errorbar(x[ok], ys[ok], xerr=[(x - lo)[ok], (hi - x)[ok]], fmt="none", ecolor=COL["et"], elinewidth=1.6, capsize=0, zorder=2)
            ax.plot(x[ok], ys[ok], ls="none", marker="s", ms=7.5, color=COL["et"], mec="white", mew=1.3, zorder=3)
            ax.axvline(0, color=MUTED, lw=1, zorder=1)
            ax.set_yticks(ys); ax.set_yticklabels(GAMES); ax.set_ylim(-0.6, len(GAMES) - 0.4)
            style(ax, f"R44, {name}", "QRC-ET − QRC (% of QRC)", "")
        F["r44_forest"] = finish(figr, axes, os.path.join(FIGDIR, "r44_forest.png")); plt.close(figr)
    return T, S, F


def texts(T, S):
    m = S["r44_mean"]
    return [
        ("md", f"# Latest | {TITLE}"),
        ("md", "**Summary.** Two checks before writing up. (1) R42 found no expected-trace gain on the paper's shared-feature task, "
               "although the paper reports one. The paper's baseline is plain TD(λ), ours is QRC, so R42 was repeated with plain "
               "TD(λ) (R42q: `--q_lambda`, no gradient correction; everything else identical). (2) The R43 MinAtar screen used 3 seeds, "
               "so the key cells were re-run on 10 fresh seeds (R44). **With TD(λ) we reproduce the paper's shared-feature result, "
               "but plain QRC is already better than TD(λ) with expected traces, and adding expected traces to QRC gives nothing more. "
               "On MinAtar, expected traces do not rescue long traces; best against best they are level.**"),
        ("md", "## Expected Traces on Top of TD(λ) and on Top of QRC (R42q against R42)\n\n"
               "Same 12 tasks, grids and seed protocol as R42 (λ and learning rate chosen on seeds 101-110, reported on 111-120), "
               "45,360 runs. Each agent with expected traces against the best agent of its own base, whole-run value error "
               "(negative = expected traces better; bold = 95% interval excludes 0):\n\n" + T["rel"]),
        ("md", f"On the paper's shared-feature task, expected traces help TD(λ) ({S['feat_td']} for 8 / 32 / 128 chains; better on "
               f"{S['feat_td_seeds']} seeds) but not QRC ({S['feat_q']}). On the one-hot tasks TD(λ) and QRC behave almost the same "
               f"(best errors within {S['oh_same']}), as expected: with one-hot features the gradient correction has little to do."),
        ("md", "### Shared-feature task: best value error of every agent\n\nThe last two columns compare QRC *without* expected traces "
               "against TD(λ) *with* expected traces, paired over the test seeds (negative = QRC better).\n\n" + T["feat"]),
        ("fig", "r42q_feat"),
        ("md", f"Plain QRC has a lower error than TD(λ) with expected traces ({S['qrc_vs_tdet_rng']}; significant in "
               f"{S['qrc_vs_tdet_sig']} comparisons). So with shared features, QRC's gradient correction appears to give the benefit "
               "that expected traces give to TD(λ), and the two do not add up."),
        ("md", "## MinAtar: 10-Seed Confirmation (R44)\n\n"
               "Settings from the R36 + R43 tuning runs (`prepare_r44.py`): for every game, QRC and QRC-ET (split form, η=0.3) at "
               "λ=0.9 and at λ=0.95 (best learning rate at that λ, θ/Q ratio 100), and each method's best setting overall. "
               "10 fresh seeds (601-610), 5M steps, 300 runs. [div.] = runs with diverged Q values, of 10.\n\n" + T["r44"]),
        ("fig", "r44_forest"),
        ("md", f"Mean over games, QRC-ET − QRC: {m.get('lam09', '–')} at λ=0.9, {m.get('lam095', '–')} at λ=0.95, "
               f"{m.get('best', '–')} best against best. Significant differences: λ=0.9: {S['r44_sig']['lam09']}; λ=0.95: "
               f"{S['r44_sig']['lam095']}; best against best: {S['r44_sig']['best']}. This confirms the R43 screen: expected traces do "
               "not make long traces usable. The single best-against-best difference (one game of five, no correction for multiple "
               "tests) should not be over-read."),
        ("md", "## Reproduction | R42q-R44\n\n" + md(["Experiment", "Job script", "Runs"], [
            ["R42q multi-chain sweep with TD(λ)", "slurm/R42_multichain_lambda.sh with `--export=ALL,BASE=q`", "45,360"],
            ["R44 MinAtar confirmation", "prepare_r44.py, then slurm/R44_minatar_confirm.sh", "300"]]) +
               "\nSummaries: `python summarize_r42.py R42q` (rows called QRC are TD(λ) there), `python summarize_r44.py`. "
               f"Full tables: `r42q_full.txt`. Code: branch [cluster-r29]({GITHUB}), `RESULTS_R42q_R44.md`."),
    ]


def write_results(items, path="RESULTS_R42q_R44.md"):
    out = ["# Expected traces with QRC: R42q and R44\n", "Generated by `wb_add_r42q_r44.py` from the cluster logs (same tables as "
           "the W&B report). γ = 0.99 everywhere.\n"]
    for kind, c in items:
        if kind == "md":
            if c.startswith("# Latest | "):
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

    RQ, nq = analyse_r42("r42_cache.pkl", "R42")
    RT, nt = analyse_r42("r42q_cache.pkl", "R42q")
    r44 = read_r44()
    print(f"complete runs: R42 {nq}, R42q {nt}, R44 {sum(len(d) for d in r44.values())}")
    T, S, F = build(RQ, RT, r44)
    for k, t in T.items():
        print(f"\n### {k}\n{t}")
    print("figures:", ", ".join(F.values()))
    items = texts(T, S)
    print("wrote", write_results(items))

    import wandb_workspaces.reports.v2 as wr
    rs = wr.Runset(entity=args.entity, project=args.project, name="Figures | R42q-R44", filters=f"Config('experiment') == '{TAG}'")
    blocks = [wr.HorizontalRule()]
    for kind, c in items:
        if kind == "md":
            blocks.append(wr.MarkdownBlock(text=c))
        elif c in F:
            blocks.append(wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title="", media_keys=[f"fig/{c}"], num_columns=1,
                                       layout=wr.Layout(x=0, y=0, w=24 if c == "r44_forest" else 16, h=11))]))
    if args.dry_run:
        print(f"\n{len(blocks)} blocks built (nothing changed online).")
        return

    rep = wr.Report.from_url(args.url)
    bl = list(rep.blocks)
    txt = [block_text(b) for b in bl]
    if any(TITLE in t for t in txt):
        raise SystemExit("The R42q-R44 section is already in the report; nothing changed.")
    i = next((k for k, t in enumerate(txt) if NEXT_SECTION in t), None)
    if i is None:
        raise SystemExit("Could not find the R42-R43 section heading; nothing changed.")
    bl[i].text = bl[i].text.replace("# Latest | ", "# Follow-up | ")
    if i > 0 and type(bl[i - 1]).__name__ == "HorizontalRule":
        i -= 1
    if not args.no_log:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, name="FOLLOW-UP | R42q-R44 report figures",
                         job_type="report-data", group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
        run.log({f"fig/{k}": wandb.Image(p) for k, p in F.items()})
        run.finish()
    backup = wr.Report.from_url(args.url)
    backup.title = backup.title + " (backup before R42q-R44)"
    backup.save(clone=True)
    print("Backup of the current report:", backup.url)
    rep.blocks = bl[:i] + blocks + bl[i:]
    rep.width = "fluid"
    rep.save()
    print(f"Inserted {len(blocks)} blocks at position {i}. Updated report:", rep.url)


if __name__ == "__main__":
    main()
