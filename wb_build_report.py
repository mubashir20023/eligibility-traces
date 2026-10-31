"""Build the W&B report for the QRC + expected-traces project from the published runs.

Needs: wb_publish_runs.py already run and synced (it writes <out>/report_data.json), the master index
markdown next to this file, and the `wandb-workspaces` package (pip install wandb-workspaces).

    python wb_build_report.py                                   # create a NEW report (recommended)
    python wb_build_report.py --append-url <existing report URL> # append everything to an existing report
    python wb_build_report.py --dry-run                         # build it, print the outline, don't save

All numbers in the R28-R31 tables come from report_data.json (computed from the run CSVs); the earlier
studies and the file index come from QRC_ET_MASTER_INDEX_AND_EVALUATIONS.md, included section by section.
"""
import argparse, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
GITHUB = "https://github.com/mubashir20023/eligibility-traces/tree/cluster-r29"
CLUSTER = "/fred/oz511/muahmed/code/stream-rep-rl"
PC_ROOT = r"C:\Users\Mubashir\stream-rep-rl"


def md_table(rows, cols, fmt=None):
    fmt = fmt or {}
    head = "| " + " | ".join(h for h, _ in cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for r in rows:
        cells = []
        for _, k in cols:
            v = r.get(k) if isinstance(k, str) else k(r)
            cells.append(fmt[k](v) if isinstance(k, str) and k in fmt and v is not None else ("–" if v is None else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def f2(v): return f"{v:.2f}"
def f4(v): return f"{v:.4f}"


def master_index_sections(path):
    """Split the master index into (heading, markdown) pairs; turn local links into readable paths."""
    text = open(path, encoding="utf-8").read()

    def fix(m):
        label, target = m.group(1), m.group(2)
        if target.startswith("http"):
            return f"[{label}]({target})"
        target = target.replace("../", "")
        return f"{label} (`{PC_ROOT}\\experiments\\{target.replace('/', chr(92))}`)" if not target.startswith(("algorithms", "README")) \
            else f"{label} (`{target}`)"
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix, text)
    parts = re.split(r"^## ", text, flags=re.M)
    intro, sections = parts[0], []
    for p in parts[1:]:
        head, _, body = p.partition("\n")
        sections.append((head.strip(), body.strip()))
    return intro, sections


def build_blocks(wr, data, entity, project):
    T, B = data["tables"], []
    cov = data["coverage"]

    def runset(name, filters, groupby=("arm", "q_lr")):
        return wr.Runset(entity=entity, project=project, name=name, filters=filters, groupby=list(groupby))

    analysis = runset("Analysis run", "Config('experiment') == 'ANALYSIS'", groupby=())

    # ---------------- summary
    fw = {(g["arm"], g["q_lr"]): g for g in T["freeway_grid"]}
    best_et = max((g for g in T["freeway_grid"] if g["arm"] == "ET eta=0.3"), key=lambda g: g["mean"], default=None)
    best_q = max((g for g in T["freeway_grid"] if g["arm"].startswith("QRC")), key=lambda g: g["mean"], default=None)
    mcb = T["multichain_best"]
    B += [wr.TableOfContents(), wr.H1(text="Bottom line"), wr.MarkdownBlock(text=(
        "* **Freeway (R28, R29; 15 fresh seeds, original Algorithm-3 code):** ET beats the usual QRC baseline "
        "(lambda 0.8), but QRC with a shorter trace (lambda 0.24) matches it. "
        + (f"Best ET {best_et['mean']:.2f} (lr {best_et['q_lr']}) vs best QRC {best_q['mean']:.2f} ({best_q['arm']}, lr {best_q['q_lr']}). " if best_et and best_q else "")
        + "The earlier Freeway gain is explained by trace length and learning rate.\n"
        "* **ET with Theta switched off diverged in all runs** at lr 0.002, while QRC with lambda 0.24 everywhere is stable: "
        "shortening only the head trace breaks it.\n"
        "* **Multi-chain prediction task (R30, R31; exact values known):** deep ET never beats the best-tuned QRC "
        + (f"({min(r['et_minus_qrc_pct'] for r in mcb):+.1f}% to {max(r['et_minus_qrc_pct'] for r in mcb):+.1f}% value error). " if mcb else "")
        + "It does beat QRC with the same lambda 0.8 under reward noise. The ordinary-trace Theta target is slightly better than "
        "Algorithm 3's recursive target, but does not close the gap.\n"
        "* **Likely reason:** best QRC's error does not grow with the number of chains, so the network's generalisation already "
        "fills in rarely visited states; in tabular prediction (where it cannot) ET has 19-31% lower error than tuned TD at 32+ chains.\n"
        "* Everything below is reproducible from the listed scripts; every run is in this project (filter by config `experiment`).")),
        wr.H2(text="Data coverage"),
        wr.MarkdownBlock(text=md_table([dict(e=e, **c) for e, c in cov.items()],
                                       [("Experiment", "e"), ("Runs published", "runs"), ("Reached full budget", "complete"),
                                        ("Log starts at step 1000", "start_at_step_1000"), ("Diverged (Q nan or >1e4)", "diverged")])
                         + "\nA run whose log does not start at step 1000 was resumed, and its CSV only holds the resumed part.")]

    # ---------------- experiment map
    B += [wr.H1(text="Experiment map (R27-R31)"), wr.MarkdownBlock(text=md_table([
        dict(id="R27", q="Freeway / Asterix at lr 0.002: ET vs QRC", d="Algorithm 3 eta 0.3 vs QRC; Freeway seeds 4-13, Asterix 1,2,3,42,999", j="18568147", s="slurm/R27_freeway_2e3.sh", sum="summarize_27.py"),
        dict(id="R28", q="Is the Freeway gain from the expected trace?", d="5 arms x 15 fresh seeds (101-115): QRC lam .8 lr 5e-4, ET eta .3 lr 2e-3, ET Theta-off, QRC lam .24 lr 5e-4 / 1e-3", j="19078432", s="slurm/R28_freeway_controls.sh", sum="summarize_r28.py"),
        dict(id="R29", q="Learning-rate sweep on the same seeds", d="QRC lam .8 at 1e-3, 2e-3; QRC lam .24 at 2e-3; ET at 5e-4, 1e-3, 4e-3", j="19146242", s="slurm/R29_freeway_lr_sweep.sh", sum="summarize_sweep.py"),
        dict(id="R30", q="Paper's multi-chain prediction task, deep agents", d="m in {1,8,32,128} x sigma in {0,1}; QRC lam {0,.24,.8}, ET eta {0,.3}; lr {3e-4,1e-3,3e-3}; seeds 1-5; --eval_values", j="19288335", s="slurm/R30_pred_screen.sh", sum="summarize_r30.py"),
        dict(id="R31", q="Can Theta learn the trace? target x Theta lr", d="ET eta 0; target e_w vs raw; et_lr {.03,.1,.3}; q_lr {3e-3,1e-2}; m in {1,32} x sigma in {0,1}", j="19290244", s="slurm/R31_theta_fix.sh", sum="summarize_r31.py"),
    ], [("ID", "id"), ("Question", "q"), ("Design", "d"), ("SLURM job", "j"), ("Job script", "s"), ("Summary script", "sum")])
        + f"\nCode: `{CLUSTER}` (cluster) and branch [cluster-r29]({GITHUB}). Agents: `algorithms/qrc.py`, `algorithms/qrc-et-algo3.py` "
          "(with the `--eval_values` patch from `apply_eval_patch.py`). Environment: `multichain_env.py`. "
          f"Run logs: `{CLUSTER}/logs/<env>__<exp_name>__<seed>/training_data.csv` and `config.json`.\n\n"
          "**Stored in this W&B project:** every run (config `experiment` = R27...R31), the ANALYSIS run (tables + figures), "
          "and the artifacts `qrc-et-cluster-code-r27-r31` (exact code and job scripts), `qrc-et-analysis-r27-r31` "
          "(report_data.json + figures) and `qrc-et-evidence-pc-experiments` (every earlier study folder from the PC).")]

    # ---------------- R28 / R29
    r28_keys = [("QRC lambda=0.8", 0.0005), ("ET eta=0.3", 0.002), ("ET eta=0.3 Theta-off", 0.002),
                ("QRC lambda=0.24", 0.0005), ("QRC lambda=0.24", 0.001)]
    pv = {c["versus"]: c for c in T["freeway_paired_vs_et_lr2e-3"]}
    rows = []
    for arm, lr in r28_keys:
        g = fw.get((arm, lr))
        if g:
            c = pv.get(f"{arm} @ lr {lr}")
            rows.append(dict(arm=arm, lr=lr, n=g["n"], mean=f"{g['mean']:.2f} ± {g['sem']:.2f}", div=g["diverged"],
                             d=(f"{c['diff']:+.2f} [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}], p={c['p']:.4f}, ET higher on {c['a_higher']}/{c['n']}" if c else "–")))
    B += [wr.H1(text="R28 | Freeway controls"), wr.MarkdownBlock(text=
          "Score = mean rolling-100-episode training return over the last 10% of 5M steps. Paired differences are "
          "ET (eta 0.3, lr 0.002) minus the arm, over the same seeds.\n\n" +
          md_table(rows, [("Arm", "arm"), ("lr", "lr"), ("n", "n"), ("Score", "mean"), ("Diverged", "div"), ("ET minus arm (95% CI)", "d")])),
          wr.PanelGrid(runsets=[runset("R28 runs", "Config('experiment') == 'R28'")], panels=[
              wr.LinePlot(title="R28 | training return (mean ± s.e. over seeds)", x="global_step", y=["avg_return"],
                          groupby_aggfunc="mean", groupby_rangefunc="stderr"),
              wr.LinePlot(title="R28 | Q-values (Theta-off runs go to NaN)", x="global_step", y=["q_values"],
                          groupby_aggfunc="mean", groupby_rangefunc="stderr")]),
          wr.PanelGrid(runsets=[analysis], panels=[wr.MediaBrowser(title="R28 learning curves", media_keys=["fig/freeway_r28_curves"], num_columns=1)])]

    lrs = sorted({g["q_lr"] for g in T["freeway_grid"]})
    arms = sorted({g["arm"] for g in T["freeway_grid"]})
    grid_rows = [dict(arm=a, **{str(lr): (f"{fw[(a, lr)]['mean']:.1f} ± {fw[(a, lr)]['sem']:.1f}" + (f" ({fw[(a, lr)]['diverged']} div.)" if fw[(a, lr)]['diverged'] else "")
                                          if (a, lr) in fw else "–") for lr in lrs}) for a in arms]
    B += [wr.H1(text="R29 | Freeway learning-rate sweep (R28 + R29, same 15 seeds)"),
          wr.MarkdownBlock(text=md_table(grid_rows, [("Agent", "arm")] + [(f"lr {lr}", str(lr)) for lr in lrs]) +
                           "\nBest ET and best QRC are statistically tied; QRC with lambda 0.24 is good across all learning rates, "
                           "ET only at 0.002. The Theta-off divergence is specific to shortening only the head trace."),
          wr.PanelGrid(runsets=[analysis], panels=[wr.MediaBrowser(title="Freeway: score vs learning rate", media_keys=["fig/freeway_lr_sweep"], num_columns=1)]),
          wr.PanelGrid(runsets=[runset("R29 runs", "Config('experiment') == 'R29'")], panels=[
              wr.LinePlot(title="R29 | training return (mean ± s.e.)", x="global_step", y=["avg_return"],
                          groupby_aggfunc="mean", groupby_rangefunc="stderr")])]

    # ---------------- R30 / R31
    B += [wr.H1(text="R30 | Multi-chain prediction task (deep agents)"), wr.MarkdownBlock(text=
          "Environment `multichain_env:MultiChainPred-m{m}-n4-s{sigma}-v0`: start -> one of m chains (random) -> shared bottleneck -> "
          "terminal reward 1 + sigma*N(0,1). One action, so the agents are value predictors. `--eval_values` logs the RMSE of the "
          "agent's values over ALL states against the exact values every 1000 steps (observations normalised with the agent's "
          "statistics, values rescaled by the agent's reward scale). AUC = mean RMSE over the 100k-step run (lower = better). "
          "Best settings are chosen on the same 5 seeds (screen).\n\n" +
          md_table(mcb, [("Task", "env"), ("Best QRC", "best_qrc"), ("QRC AUC", "best_qrc_auc"), ("Best ET", "best_et"),
                         ("ET AUC", "best_et_auc"), ("ET vs QRC", "et_minus_qrc_pct"), ("ET lower on", lambda r: f"{r['et_lower_on']}/{r['n']}")],
                   {"best_qrc_auc": f4, "best_et_auc": f4, "et_minus_qrc_pct": lambda v: f"{v:+.1f}%"})),
          wr.PanelGrid(runsets=[analysis], panels=[
              wr.MediaBrowser(title="Value error vs number of chains", media_keys=["fig/multichain_vs_chains"], num_columns=1),
              wr.MediaBrowser(title="m=32, sigma=1 learning curves", media_keys=["fig/multichain_m32_s1_curves"], num_columns=1)]),
          wr.PanelGrid(runsets=[runset("R30 m=32 sigma=1", "Config('experiment') == 'R30' and Config('chains') == 32 and Config('reward_noise') == 1.0")],
                       panels=[wr.LinePlot(title="R30 | m=32, sigma=1 | value RMSE (mean ± s.e.)", x="global_step", y=["value_rmse"],
                                           groupby_aggfunc="mean", groupby_rangefunc="stderr")])]
    allrows = T["multichain_all"]
    for env in [r["env"] for r in mcb] or sorted({r["env"] for r in allrows}):
        rows = sorted([r for r in allrows if r["env"] == env], key=lambda r: r["auc"])
        B += [wr.H3(text=f"All settings | {env}"),
              wr.MarkdownBlock(text=md_table(rows, [("Exp.", "experiment"), ("Agent", "arm"), ("q_lr", "q_lr"), ("n", "n"),
                                                    ("Value-error AUC", "auc"), ("s.e.", "sem"), ("Final trace loss", "trace_loss")],
                                             {"auc": f4, "sem": f4, "trace_loss": f4}))]
    B += [wr.H1(text="R31 | Theta target and learning rate"), wr.MarkdownBlock(text=
          "R31 rows appear in the tables above as 'ET eta=0.0 target=raw ...' (ordinary trace target) and 'ET eta=0.0 Theta-lr=...' "
          "(Algorithm-3 e_w target). Findings: the ordinary-trace target is slightly better in every task and fits the trace "
          "more accurately; Q learning rate 1e-2 is too high with noisy rewards; no setting beats the best QRC."),
          wr.PanelGrid(runsets=[runset("R31 m=32 sigma=1", "Config('experiment') == 'R31' and Config('chains') == 32 and Config('reward_noise') == 1.0",
                                       groupby=("arm", "q_lr"))],
                       panels=[wr.LinePlot(title="R31 | m=32, sigma=1 | value RMSE", x="global_step", y=["value_rmse"],
                                           groupby_aggfunc="mean", groupby_rangefunc="stderr"),
                               wr.LinePlot(title="R31 | m=32, sigma=1 | Theta trace loss", x="global_step", y=["et_trace_loss"],
                                           groupby_aggfunc="mean", groupby_rangefunc="stderr", log_y=True)])]

    # ---------------- tabular checks
    B += [wr.H1(text="Local tabular checks (numpy, 23-26 Oct)"), wr.MarkdownBlock(text=
          "Scripts: `experiments/tabular_checks/` in the repo (also `multichain_env.py` in the repo root).\n\n"
          "| Check | Script | Result |\n|---|---|---|\n"
          "| Prediction multi-chain, Theta step 1/n vs fixed | `prediction_multichain.py` | ET (1/n) beats TD(lambda) and damping at every m; fixed-step ET loses at m=128 (0.320 vs 0.242) |\n"
          "| Same, TD lambda tuned incl. 0 | `prediction_multichain_tuned_lambda.py` | m=32: 0.067 vs 0.083; m=128: 0.166 vs 0.242 (ET 19-31% lower); m=8 tie |\n"
          "| Control, decision at start | `control_start_decision.py` | No ET benefit (e.g. m=16, sigma=1: 78% vs 66-77%) |\n"
          "| Control, quit anywhere | `control_quit_anywhere.py` | All methods quit (return ~0.5); uninformative |\n"
          "| Control, quit at chain end | `control_quit_at_chain_end.py` | ET beats Q(lambda=0.8) with noise (m=8, sigma=1: 0.906 vs 0.824) |\n"
          "| Same, lambda tuned | `control_quit_at_chain_end_tuned_lambda.py` | Tie in 5 of 6 settings (best Q usually lambda=0) |\n")]
    return B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--data", default="wandb_upload/report_data.json")
    ap.add_argument("--master-index", default=os.path.join(HERE, "QRC_ET_MASTER_INDEX_AND_EVALUATIONS.md"))
    ap.add_argument("--title", default="QRC and Expected Traces | Complete Results R27-R31 and Evidence Index")
    ap.add_argument("--append-url", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import wandb_workspaces.reports.v2 as wr
    data = json.load(open(args.data))
    blocks = build_blocks(wr, data, args.entity, args.project)
    if os.path.exists(args.master_index):
        intro, sections = master_index_sections(args.master_index)
        blocks += [wr.H1(text="Earlier studies and full evidence index (master index, 23 Oct 2026)"),
                   wr.MarkdownBlock(text="Copied from `QRC_ET_MASTER_INDEX_AND_EVALUATIONS.md`. Paths are on the PC "
                                         f"(`{PC_ROOT}`) unless marked as cluster paths.\n\n" + intro.split("\n", 1)[-1].strip())]
        for head, body in sections:
            blocks += [wr.H2(text=head), wr.MarkdownBlock(text=body)]
    else:
        print(f"WARNING: master index not found at {args.master_index}; earlier studies section skipped")
    blocks += [wr.H1(text="Reproduce"), wr.CodeBlock(code=(
        "# cluster, from " + CLUSTER + "\n"
        "sbatch slurm/R28_freeway_controls.sh; sbatch slurm/R29_freeway_lr_sweep.sh\n"
        "python apply_eval_patch.py            # adds --eval_values to both agents (once)\n"
        "sbatch slurm/R30_pred_screen.sh; sbatch slurm/R31_theta_fix.sh\n"
        "python summarize_r28.py; python summarize_sweep.py; python summarize_r30.py; python summarize_r31.py\n"
        "python wb_publish_runs.py && (cd wandb_upload && wandb sync --sync-all) && python wb_build_report.py"), language="python")]

    print(f"{len(blocks)} blocks built")
    if args.dry_run:
        for b in blocks:
            if type(b).__name__ in ("H1", "H2"):
                print(f"  {type(b).__name__}: {b.text}")
        return
    if args.append_url:
        report = wr.Report.from_url(args.append_url)
        report.blocks = list(report.blocks) + [wr.HorizontalRule()] + blocks
    else:
        report = wr.Report(project=args.project, entity=args.entity, title=args.title, width="fluid",
                           description="Every QRC / QRC-ET test: R27-R31 runs, multi-chain controls, tabular checks, earlier studies and file locations.",
                           blocks=blocks)
    report.save()
    print("Report saved:", report.url)


if __name__ == "__main__":
    main()
