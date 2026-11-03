"""Add the R36 (equal-budget tuning) + R37 (best settings on fresh seeds) section to the existing report, as the
newest follow-up: inserted right before the R33-R35 section, same layout (compact tables + coloured figure pairs).

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_add_r36_r37.py --dry-run     # build tables + figures locally, change nothing
    ~/wbenv/bin/python wb_add_r36_r37.py               # log figures run, back up the report, insert, save
"""
import argparse, csv, glob, os, re
from collections import defaultdict

import numpy as np

from wb_publish_runs import paired, mean_sem
from wb_reorganize import fig_pair, md, block_text, REPORT_URL, GITHUB

TAG = "FOLLOWUP-R36-R37-FIGS"
FIGDIR = "report_figures"
GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]
TOTAL = 5_000_000
NEXT_SECTION = "Expected Traces with the Fixes (R33-R35)"   # heading of the section we insert in front of


def read(prefix):
    """{(game, name): {seed: dict(score, diverged, step, ret)}} for complete runs."""
    out = defaultdict(dict)
    for p in glob.glob(f"logs/**/*-v1__{prefix}_*__*/training_data.csv", recursive=True):
        m = re.search(rf"(\w+)-v1__{prefix}_(.+)__(\d+)/", p)
        rows = list(csv.DictReader(open(p))) if m else []
        if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
            continue
        st = np.array([float(r["global_step"]) for r in rows]); ret = np.array([float(r["avg_return"]) for r in rows])
        q = np.array([float(r["q_values"]) for r in rows])
        out[(m.group(1), m.group(2))][int(m.group(3))] = dict(
            score=float(ret[st > 0.9 * TOTAL].mean()), step=st, ret=ret,
            diverged=bool(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4))
    return out


def parse36(cfg):
    m = re.match(r"(qrc|et)_lam([\d.]+)(?:_r(\d+))?_lr([\de.-]+)$", cfg)
    return (m.group(1), m.group(2), m.group(3), m.group(4)) if m else None


def nice36(cfg):
    k, lam, r, lr = parse36(cfg)
    return f"λ={lam}, lr {lr}" + (f", ratio {r}" if r else "")


def build(R36, R37):
    T, F = {}, {}
    mean = lambda d: float(np.mean([v["score"] for v in d.values()]))
    # ---------------- R36 tuning
    best36, div, ratio_tab = {}, defaultdict(lambda: [0, 0]), []
    lam_curve = {"qrc": defaultdict(list), "et": defaultdict(list)}
    ratio_curve = {}
    for g in GAMES:
        for meth in ("qrc", "et"):
            cfgs = [(mean(d), c) for (gg, c), d in R36.items() if gg == g and parse36(c) and parse36(c)[0] == meth]
            if cfgs:
                best36[(g, meth)] = max(cfgs)
            div[(g, meth)] = [sum(v["diverged"] for (gg, c), d in R36.items() if gg == g and parse36(c) and parse36(c)[0] == meth
                                  for v in d.values()),
                              sum(len(d) for (gg, c), d in R36.items() if gg == g and parse36(c) and parse36(c)[0] == meth)]
        if (g, "qrc") not in best36:
            continue
        ref = best36[(g, "qrc")][0]
        for meth in ("qrc", "et"):                                    # best score per lambda, relative to best QRC
            by = defaultdict(list)
            for (gg, c), d in R36.items():
                if gg == g and parse36(c) and parse36(c)[0] == meth:
                    by[float(parse36(c)[1])].append(mean(d))
            for lam, v in by.items():
                lam_curve[meth][lam].append(max(v) / ref)
        by = defaultdict(list)
        for (gg, c), d in R36.items():
            if gg == g and parse36(c) and parse36(c)[0] == "et":
                by[int(parse36(c)[2])].append(mean(d))
        if by:
            rs = sorted(by)
            ratio_curve[g] = (rs, [max(by[r]) / ref for r in rs])
            ratio_tab.append([g] + [f"{max(by[r]):.1f} / {np.median(by[r]):.1f}" for r in rs])
    rows = []
    for g in GAMES:
        if (g, "qrc") in best36 and (g, "et") in best36:
            q, e = best36[(g, "qrc")], best36[(g, "et")]
            rows.append([g, f"{q[0]:.1f} ({nice36(q[1])})", f"{e[0]:.1f} ({nice36(e[1])})", f"{e[0] - q[0]:+.1f}",
                         f"{div[(g, 'qrc')][0]} / {div[(g, 'et')][0]}"])
    T["tune"] = md(["Game", "Best QRC (setting)", "Best QRC-ET (setting)", "ET − QRC", "Diverged runs QRC / ET (of 72 each)"], rows)
    T["ratio"] = md(["Game", "ratio 30", "ratio 100", "ratio 300"], ratio_tab) if ratio_tab else ""
    tot = [sum(div[(g, m)][0] for g in GAMES) for m in ("qrc", "et")]
    n = [sum(div[(g, m)][1] for g in GAMES) for m in ("qrc", "et")]

    # ---------------- R37 final
    rows, curves = [], {}
    for g in GAMES:
        q, e = R37.get((g, next((c for gg, c in R37 if gg == g and c.startswith("qrc_best")), ""))), \
               R37.get((g, next((c for gg, c in R37 if gg == g and c.startswith("et_best")), "")))
        if not q or not e:
            continue
        qs, es = {s: v["score"] for s, v in q.items()}, {s: v["score"] for s, v in e.items()}
        c = paired(es, qs)
        (qm, qse), (em, ese) = mean_sem(qs.values()), mean_sem(es.values())
        bq, be = best36.get((g, "qrc"), (0, "")), best36.get((g, "et"), (0, ""))
        rows.append([g, f"{qm:.1f} ± {qse:.1f}", nice36(bq[1]) if bq[1] else "–", f"{em:.1f} ± {ese:.1f}",
                     nice36(be[1]) if be[1] else "–", f"{c['diff']:+.1f} [{c['ci_low']:+.1f}, {c['ci_high']:+.1f}]",
                     f"{c['a_higher']}/{c['n']}"])
        gx = np.arange(25_000, TOTAL + 1, 25_000)
        curves[g] = {"x": gx, "best QRC": np.mean([np.interp(gx, v["step"], v["ret"]) for v in q.values()], 0),
                     "best QRC-ET (split, η=0.3)": np.mean([np.interp(gx, v["step"], v["ret"]) for v in e.values()], 0)}
    T["final"] = md(["Game", "Best QRC score", "QRC setting", "Best QRC-ET score", "QRC-ET setting",
                     "ET − QRC (95% CI)", "ET higher on"], rows)
    T["design"] = md(["Method", "Learning rates", "λ", "Θ / Q learning-rate ratio", "Settings"], [
        ["QRC", "2.5e-4, 5e-4, 1e-3, 2e-3", "0, 0.24, 0.5, 0.8, 0.9, 0.95", "–", "24"],
        ["QRC-ET (split form, no lag, η=0.3)", "2.5e-4, 5e-4, 1e-3, 2e-3", "0.24, 0.8", "30, 100, 300", "24"]])
    T["repro"] = md(["Experiment", "SLURM job", "Job script", "Runs"], [
        ["R36 equal-budget tuning (seeds 301-303)", "19651125", "slurm/R36_minatar_tune.sh", 720],
        ["R37 best settings, fresh seeds 401-410", "19802608", "slurm/R37_minatar_final.sh", 100]])

    # ---------------- figures
    os.makedirs(FIGDIR, exist_ok=True)
    ret = ("Environment steps", "rolling-100-episode return")
    panel = lambda g: (f"{g}: best settings, 10 fresh seeds", curves[g]["x"],
                       {k: v for k, v in curves[g].items() if k != "x"}, *ret, {})
    for key, gs in [("r37_as_br", ["Asterix", "Breakout"]), ("r37_fw_sq", ["Freeway", "Seaquest"])]:
        if all(g in curves for g in gs):
            F[key] = fig_pair(os.path.join(FIGDIR, key + ".png"), [panel(g) for g in gs])
    if "SpaceInvaders" in curves and ratio_curve:
        rs = next(iter(ratio_curve.values()))[0]
        F["r37_si_ratio"] = fig_pair(os.path.join(FIGDIR, "r37_si_ratio.png"), [
            panel("SpaceInvaders"),
            ("R36: best QRC-ET score vs Θ/Q ratio (÷ best QRC)", np.array(rs, float),
             {g: v for g, (_, v) in ratio_curve.items()},
             "Θ / Q learning-rate ratio", "best score ÷ best QRC score", {"logx": True, "markers": True, "xticks": rs})])
    lam_q = sorted(lam_curve["qrc"]); lam_e = sorted(lam_curve["et"])
    if lam_q:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9))
        ax = axes[0]
        ax.plot(lam_q, [np.mean(lam_curve["qrc"][l]) for l in lam_q], "o-", color="#1f77b4", label="QRC")
        ax.plot(lam_e, [np.mean(lam_curve["et"][l]) for l in lam_e], "s-", color="#d62728", label="QRC-ET (split, η=0.3)")
        ax.set_title("R36: best score at each λ (÷ best QRC, mean over 5 games)", fontsize=11)
        ax.set_xlabel("λ"); ax.set_ylabel("best score ÷ best QRC score"); ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False)
        ax = axes[1]
        x = np.arange(len(GAMES))
        ax.bar(x - 0.2, [div[(g, "qrc")][0] for g in GAMES], 0.4, color="#1f77b4", label="QRC")
        ax.bar(x + 0.2, [div[(g, "et")][0] for g in GAMES], 0.4, color="#d62728", label="QRC-ET (split, η=0.3)")
        ax.set_xticks(x); ax.set_xticklabels(GAMES, fontsize=9)
        ax.set_title(f"R36: diverged runs (QRC {tot[0]}/{n[0]}, QRC-ET {tot[1]}/{n[1]})", fontsize=11)
        ax.set_ylabel("diverged runs"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); F["r36_lambda_div"] = os.path.join(FIGDIR, "r36_lambda_div.png"); fig.savefig(F["r36_lambda_div"], dpi=150); plt.close(fig)
    return T, F, tot, n


def blocks(wr, T, F, tot, n, entity, project):
    rs = wr.Runset(entity=entity, project=project, name="Figures | R36-R37", filters=f"Config('experiment') == '{TAG}'")
    M = lambda t: wr.MarkdownBlock(text=t)

    def fig(key, h=11):
        return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title="", media_keys=[f"fig/{key}"], num_columns=1,
                                                                  layout=wr.Layout(x=0, y=0, w=24, h=h))]) if key in F else None
    out = [
        wr.HorizontalRule(),
        M("# Latest | Equal-Budget Tuning on All Five MinAtar Games (R36, R37)"),
        M("**Summary.** Peter's question: with the core algorithm working on multi-chain, is there a MinAtar setting where "
          "QRC-ET does better, for example a different ratio between Θ's and Q's learning rates? Both methods got the same "
          "tuning budget, and each method's best setting per game was then re-run on 10 fresh seeds. **Result: QRC-ET is "
          "level with QRC on all five games (no significant difference on any).** The Θ/Q ratio made little difference. "
          f"QRC-ET was more robust: {tot[1]} of {n[1]} tuning runs diverged, against {tot[0]} of {n[0]} for QRC."),
        M("## Final Comparison | Best Settings on 10 Fresh Seeds (R37)\n\n"
          "Final score = rolling-100 return over the last 10% of 5M steps, mean ± s.e.; ET − QRC is paired over seeds 401-410.\n\n"
          + T["final"]),
        fig("r37_as_br"),
        fig("r37_fw_sq"),
        fig("r37_si_ratio"),
        M("## Tuning Grid | Same Budget for Both Methods (R36)\n\n" + T["design"] +
          "\nAll five games, 3 tuning seeds (301-303), 5M steps, 720 runs. The Θ/Q ratio sets Θ's learning rate as a "
          "multiple of Q's (the earlier default, Θ lr 0.1 at Q lr 1e-3, is ratio 100)."),
        M("## Tuning Results (R36, 3 seeds)\n\nBest of 24 settings per method; optimistic for both, so stage 2 (R37) uses "
          "fresh seeds.\n\n" + T["tune"]),
        M("### QRC-ET by Θ/Q Ratio (best / median score over learning rates and λ)\n\n" + T["ratio"] +
          "\nMaking Θ adapt faster (ratio 300) did not help; the ratio is not the bottleneck."),
        fig("r36_lambda_div"),
        M("QRC-ET's best λ was 0.24 (the shortest it was offered) on 4 of 5 games, while QRC often preferred λ=0.5, which "
          "QRC-ET did not get; if anything this favours QRC."),
        M("## Reproduction | R36-R37\n\n" + T["repro"] +
          f"\nCode: branch [cluster-r29]({GITHUB}). All 820 runs are in this project, tagged R36 / R37."),
    ]
    return [b for b in out if b is not None]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=REPORT_URL)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    args.url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]
    import wandb_workspaces.reports.v2 as wr

    R36, R37 = read("R36"), read("R37")
    print(f"complete runs: R36 {sum(len(d) for d in R36.values())}, R37 {sum(len(d) for d in R37.values())}")
    T, F, tot, n = build(R36, R37)
    for k, v in T.items():
        print(f"\n### {k}\n{v}")
    print("figures:", ", ".join(F.values()))
    new = blocks(wr, T, F, tot, n, args.entity, args.project)
    if args.dry_run:
        print(f"\n{len(new)} blocks built (nothing changed).")
        return

    rep = wr.Report.from_url(args.url)
    bl = list(rep.blocks)
    texts = [block_text(b) for b in bl]
    if any("Equal-Budget Tuning on All Five MinAtar Games" in t for t in texts):
        raise SystemExit("The R36-R37 section is already in the report; nothing changed.")
    i = next((k for k, t in enumerate(texts) if NEXT_SECTION in t), None)
    if i is None:
        raise SystemExit("Could not find the R33-R35 section heading; nothing changed.")
    bl[i].text = bl[i].text.replace("# Latest | ", "# Follow-up | ")       # no longer the latest
    if i > 0 and type(bl[i - 1]).__name__ == "HorizontalRule":
        i -= 1
    if not args.no_log:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, name="FOLLOW-UP | R36-R37 report figures",
                         job_type="report-data", group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
        run.log({f"fig/{k}": wandb.Image(p) for k, p in F.items()})
        run.finish()
    backup = wr.Report.from_url(args.url)
    backup.title = backup.title + " (backup before R36-R37)"
    backup.save(clone=True)
    print("Backup of the current report:", backup.url)
    rep.blocks = bl[:i] + new + bl[i:]
    rep.width = "fluid"
    rep.save()
    print(f"Inserted {len(new)} blocks at position {i}. Updated report:", rep.url)


if __name__ == "__main__":
    main()
