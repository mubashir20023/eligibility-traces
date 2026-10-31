"""Append the R28-R31 follow-up to the EXISTING five-game report, in the same format:
short paragraphs, one compact "Report tables" panel per section, then side-by-side comparison plots.

Steps it performs:
  1. Reads every R28-R31 run CSV (same code as wb_publish_runs.py) and computes mean curves + compact tables.
  2. Logs them to one small online run "FOLLOW-UP | R28-R31 report data" (tables + plot series).
  3. Saves a BACKUP copy of the existing report, then appends the new sections to the original and saves it.

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_append_report.py --inspect        # print the existing report's structure, change nothing
    ~/wbenv/bin/python wb_append_report.py --dry-run        # build everything, log nothing, change nothing
    ~/wbenv/bin/python wb_append_report.py                  # log data run, back up, append, save
"""
import argparse, json, os
from collections import defaultdict

import numpy as np

from wb_publish_runs import discover, read_run, run_info, run_summary, paired, mean_sem

REPORT_URL = ("https://forge.coreweave.com/mubbashirahmed/qrc-et-algo3-minatar/reports/"
              "QRC-vs-Experimental-QRC-ET-%7C-Five-Game-Screen-and-Breakout-Follow-up--VmlldzoxODExNTc2Nw==")
TAG = "FOLLOWUP-R28-R31"
GITHUB = "https://github.com/mubashir20023/eligibility-traces/tree/cluster-r29"
ET, ETOFF = "ET eta=0.3", "ET eta=0.3 Theta-off"
Q08, Q024, Q0 = "QRC lambda=0.8", "QRC lambda=0.24", "QRC lambda=0.0"
NICE = {ET: "QRC-ET (η=0.3)", ETOFF: "QRC-ET, Θ off", Q08: "QRC (λ=0.8)", Q024: "QRC (λ=0.24)", Q0: "QRC (λ=0)"}


def nice(arm):
    if arm in NICE:
        return NICE[arm]
    return arm.replace("ET eta=", "QRC-ET η=").replace("QRC lambda=", "QRC λ=").replace("Theta-lr", "Θ lr")


# ------------------------------------------------------------------------------------------------ data
def load_runs():
    runs = []
    for p in discover(["R28", "R29", "R30", "R31"]):
        parsed = run_info(p)
        if parsed:
            info, cfg = parsed
            cols = read_run(p)
            s = run_summary(cols, cfg.get("total_timesteps"))
            if s.get("complete"):
                runs.append(dict(info=info, cols=cols, summ=s))
    return runs


def mean_curve(runs, key, grid):
    ys = [np.interp(grid, r["cols"]["global_step"], np.nan_to_num(r["cols"][key], nan=0.0 if key == "avg_return" else 10.0))
          for r in runs]
    return np.mean(ys, 0) if ys else None


def build_data(runs):
    D = {"tables": {}, "series": {}}
    fw = defaultdict(list)
    for r in runs:
        if r["info"]["experiment"] in ("R28", "R29"):
            fw[(r["info"]["arm"], r["info"]["q_lr"])].append(r)
    score = {k: {r["info"]["seed"]: r["summ"]["final_score"] for r in v} for k, v in fw.items()}

    # --- Freeway main table (R28 arms + best QRC lambda 0.24)
    ref = score.get((ET, 0.002), {})
    rows = []
    for arm, lr in [(ET, 0.002), (Q08, 0.0005), (Q024, 0.002), (Q024, 0.001), (ETOFF, 0.002)]:
        if (arm, lr) not in score:
            continue
        mu, se = mean_sem(score[(arm, lr)].values())
        c = paired(ref, score[(arm, lr)]) if (arm, lr) != (ET, 0.002) else None
        rows.append([nice(arm), lr, len(score[(arm, lr)]), f"{mu:.2f} ± {se:.2f}",
                     f"{c['diff']:+.2f}" if c else "–", f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]" if c else "–",
                     f"{c['a_higher']}/{c['n']}" if c else "–"])
    D["tables"]["Freeway | 15 Fresh Seeds"] = (["Agent", "Learning rate", "Seeds", "Score", "ET − agent", "95% CI", "ET higher on"], rows)

    # --- Freeway learning-rate grid
    lrs = [0.0005, 0.001, 0.002, 0.004]
    grid_rows = []
    for arm in [ET, Q08, Q024]:
        cells = []
        for lr in lrs:
            if (arm, lr) in score:
                mu, se = mean_sem(score[(arm, lr)].values())
                div = sum(r["summ"].get("diverged", False) for r in fw[(arm, lr)])
                cells.append(f"{mu:.1f} ± {se:.1f}" + (f" ({div} diverged)" if div else ""))
            else:
                cells.append("–")
        grid_rows.append([nice(arm)] + cells)
    D["tables"]["Freeway | Score by Learning Rate"] = (["Agent"] + [f"lr {lr}" for lr in lrs], grid_rows)

    # --- Freeway curves (mean over 15 seeds), every 25k steps
    g = np.arange(25_000, 5_000_001, 25_000)
    D["series"]["fw"] = {"x": g.tolist()}
    for key, (arm, lr) in {"et": (ET, 0.002), "qrc08": (Q08, 0.0005), "qrc024": (Q024, 0.002), "etoff": (ETOFF, 0.002)}.items():
        if (arm, lr) in fw:
            D["series"]["fw"][key] = mean_curve(fw[(arm, lr)], "avg_return", g).tolist()
    D["series"]["lr"] = {"x": lrs}
    for key, arm in {"et": ET, "qrc08": Q08, "qrc024": Q024}.items():
        D["series"]["lr"][key] = [float(np.mean(list(score[(arm, lr)].values()))) if (arm, lr) in score else None for lr in lrs]

    # --- Multi-chain
    mc = defaultdict(list)
    for r in runs:
        i = r["info"]
        if i["experiment"] in ("R30", "R31") and "value_rmse_auc" in r["summ"]:
            mc[(i["chains"], i["reward_noise"], i["experiment"], i["arm"], i["q_lr"])].append(r)
    auc = {k: {r["info"]["seed"]: r["summ"]["value_rmse_auc"] for r in v} for k, v in mc.items()}

    def best(m, s, pred):
        c = [(np.mean(list(v.values())), k) for k, v in auc.items() if k[0] == m and k[1] == s and pred(k) and len(v) >= 5]
        return min(c) if c else (None, None)

    rows, chains = [], {}
    for s in (0.0, 1.0):
        for m in (1, 8, 32, 128):
            qa, qk = best(m, s, lambda k: k[3].startswith("QRC"))
            ea, ek = best(m, s, lambda k: k[3].startswith("ET"))
            q8, _ = best(m, s, lambda k: k[3] == Q08)
            if qk is None or ek is None:
                continue
            c = paired(auc[ek], auc[qk])
            rows.append([m, s, f"{qa:.4f}", nice(qk[3]) + f", lr {qk[4]}", f"{ea:.4f}", nice(ek[3]) + f", lr {ek[4]}",
                         f"{100 * (ea / qa - 1):+.1f}%", f"{c['n'] - c['a_higher']}/{c['n']}" if c else "–"])
            chains[(m, s)] = dict(best_qrc=qa, qrc08=q8, best_et=ea, qk=qk, ek=ek)
    D["tables"]["Multi-Chain | Value Error (lower is better)"] = (
        ["Chains", "Reward noise σ", "Best QRC error", "Best QRC setting", "Best ET error", "Best ET setting", "ET vs QRC", "ET better on"], rows)
    for s in (0.0, 1.0):
        tag = f"m_s{int(s)}"
        D["series"][tag] = {"x": [m for m in (1, 8, 32, 128) if (m, s) in chains]}
        for key in ("best_qrc", "qrc08", "best_et"):
            D["series"][tag][key] = [chains[(m, s)][key] for m in D["series"][tag]["x"]]
    gx = np.arange(1000, 100_001, 1000)
    for m in (1, 32, 128):
        if (m, 1.0) in chains:
            c = chains[(m, 1.0)]
            q08k = min(((np.mean(list(auc[k].values())), k) for k in auc if k[0] == m and k[1] == 1.0 and k[3] == Q08), default=(None, None))[1]
            D["series"][f"mc{m}"] = {"x": gx.tolist(), "best_qrc": mean_curve(mc[c["qk"]], "value_rmse", gx).tolist(),
                                     "best_et": mean_curve(mc[c["ek"]], "value_rmse", gx).tolist()}
            if q08k:
                D["series"][f"mc{m}"]["qrc08"] = mean_curve(mc[q08k], "value_rmse", gx).tolist()

    # --- R31 Theta target x Theta lr (m=32, sigma=1, q_lr 3e-3)
    base_q = chains.get((32, 1.0), {}).get("best_qrc")
    rows, tl = [], {"x": gx.tolist()}
    for tgt, tname in [("raw", "ordinary trace"), ("e_w", "Algorithm 3 (e_w)")]:
        for etlr in (0.03, 0.1, 0.3):
            k = [k for k in mc if k[0] == 32 and k[1] == 1.0 and k[2] == "R31" and k[4] == 0.003
                 and (("target=raw" in k[3]) == (tgt == "raw")) and ((f"Theta-lr={etlr}" in k[3]) or (etlr == 0.1 and "Theta-lr" not in k[3]))]
            if not k:
                continue
            rs = mc[k[0]]
            a = np.mean([r["summ"]["value_rmse_auc"] for r in rs])
            t = np.mean([r["summ"].get("trace_loss_final", np.nan) for r in rs])
            rows.append([tname, etlr, f"{a:.4f}", f"{100 * (a / base_q - 1):+.1f}%" if base_q else "–", f"{t:.4f}"])
            if etlr == 0.1:
                tl[tgt] = mean_curve(rs, "et_trace_loss", gx).tolist()
    D["tables"]["Θ Target and Learning Rate | m=32, σ=1"] = (["Θ target", "Θ learning rate", "Value error", "vs best QRC", "Final trace loss"], rows)
    D["series"]["tl"] = tl

    D["tables"]["Reproduction | R28-R31"] = (["Experiment", "SLURM job", "Job script", "Runs"], [
        ["R28 Freeway controls", "19078432", "slurm/R28_freeway_controls.sh", 75],
        ["R29 Freeway learning-rate sweep", "19146242", "slurm/R29_freeway_lr_sweep.sh", 90],
        ["R30 Multi-chain prediction screen", "19288335", "slurm/R30_pred_screen.sh", 600],
        ["R31 Θ target / learning rate", "19290244", "slurm/R31_theta_fix.sh", 240]])
    return D


# ------------------------------------------------------------------------------------------------ data run
def log_data_run(D, entity, project):
    import wandb
    run = wandb.init(entity=entity, project=project, name="FOLLOW-UP | R28-R31 report data", job_type="report-data",
                     group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
    S = D["series"]
    for prefix, step in [("fw", "fw_step"), ("lr", "lr_x"), ("m_s0", "m_x0"), ("m_s1", "m_x1"),
                         ("mc1", "mc_step"), ("mc32", "mc_step"), ("mc128", "mc_step"), ("tl", "mc_step")]:
        wandb.define_metric(step)
        wandb.define_metric(f"{prefix}/*", step_metric=step)
    for prefix, step in [("fw", "fw_step"), ("lr", "lr_x"), ("m_s0", "m_x0"), ("m_s1", "m_x1"),
                         ("mc1", "mc_step"), ("mc32", "mc_step"), ("mc128", "mc_step"), ("tl", "mc_step")]:
        s = S.get(prefix)
        if not s:
            continue
        for j, x in enumerate(s["x"]):
            row = {step: x}
            for k, v in s.items():
                if k != "x" and v[j] is not None:
                    row[f"{prefix}/{k}"] = v[j]
            wandb.log(row)
    for name, (cols, rows) in D["tables"].items():
        wandb.log({name: wandb.Table(columns=cols, data=[[str(c) for c in r] for r in rows])})
    run.finish()


# ------------------------------------------------------------------------------------------------ report blocks
def build_blocks(wr, D, entity, project, table_panel):
    rs = wr.Runset(entity=entity, project=project, name="Report tables | R28-R31", filters=f"Config('experiment') == '{TAG}'")
    T = D["tables"]

    def table(name, h=8):
        if table_panel == "media":
            return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title=name, media_keys=[name], num_columns=1,
                                                                      layout=wr.Layout(x=0, y=0, w=24, h=h))])
        if table_panel == "markdown":
            cols, rows = T[name]
            md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)
            return wr.MarkdownBlock(text=f"**{name}**\n\n" + md)
        return wr.PanelGrid(runsets=[rs], panels=[wr.WeavePanelSummaryTable(table_name=name, layout=wr.Layout(x=0, y=0, w=24, h=h))])

    def pair(left, right, h=9):
        panels = []
        for i, p in enumerate([left, right]):
            if p:
                title, x, ys, titles, extra = p
                panels.append(wr.LinePlot(title=title, x=x, y=ys, line_titles=titles, layout=wr.Layout(x=12 * i, y=0, w=12, h=h), **extra))
        return wr.PanelGrid(runsets=[rs], panels=panels)

    def MD(text):
        return wr.MarkdownBlock(text=text)

    fwT = {r[0]: r for r in T["Freeway | 15 Fresh Seeds"][1]}
    mcT = T["Multi-Chain | Value Error (lower is better)"][1]
    noisy = [r for r in mcT if r[1] == 1.0]
    ret = {"title_y": "Rolling-100-episode training return", "title_x": "Environment steps"}
    B = [wr.HorizontalRule(),
         MD("# Follow-up | Freeway Controls (R28, R29)"),
         MD("15 fresh seeds (101-115), original Algorithm-3 agent at the R27 setting, 5M steps. "
                   "Score is the final-10% rolling return, as above. The question: does the R27 Freeway gain come from the expected trace?"),
         table("Freeway | 15 Fresh Seeds", h=8),
         MD("QRC-ET beats the usual QRC baseline (λ=0.8), but QRC with a shorter trace (λ=0.24) matches it within noise. "
                   "With Θ switched off every run diverged."),
         pair(("Freeway | QRC-ET vs QRC (λ=0.8)", "fw_step", ["fw/et", "fw/qrc08"],
               {"fw/et": "QRC-ET (η=0.3, lr 0.002)", "fw/qrc08": "QRC (λ=0.8, lr 0.0005)"}, ret),
              ("Freeway | QRC-ET vs QRC (λ=0.24)", "fw_step", ["fw/et", "fw/qrc024"],
               {"fw/et": "QRC-ET (η=0.3, lr 0.002)", "fw/qrc024": "QRC (λ=0.24, lr 0.002)"}, ret)),
         MD("## Freeway | Learning-Rate Sweep"),
         MD("Same 15 seeds. QRC with λ=0.24 is good at every rate; QRC-ET only at 0.002."),
         table("Freeway | Score by Learning Rate", h=6),
         pair(("Freeway | Final score vs learning rate", "lr_x", ["lr/et", "lr/qrc08", "lr/qrc024"],
               {"lr/et": "QRC-ET (η=0.3)", "lr/qrc08": "QRC (λ=0.8)", "lr/qrc024": "QRC (λ=0.24)"},
               {"log_x": True, "title_x": "Q learning rate", "title_y": "Final score"}),
              ("Freeway | QRC-ET vs QRC-ET with Θ off", "fw_step", ["fw/et", "fw/etoff"],
               {"fw/et": "QRC-ET", "fw/etoff": "QRC-ET, Θ off (diverges)"}, ret)),
         MD("# Follow-up | Multi-Chain Prediction Test (R30, R31)"),
         MD("The expected-traces paper's multi-chain task, run with the same deep agents: m random chains merge into one state, "
                   "terminal reward 1 + σ·noise, exact values known. Metric: value error over all states (lower is better), 5 seeds, "
                   "best learning rate and λ per agent."),
         table("Multi-Chain | Value Error (lower is better)", h=11),
         MD(f"QRC-ET never beats the best-tuned QRC ({min(float(r[6][:-1]) for r in noisy):+.1f}% to "
                   f"{max(float(r[6][:-1]) for r in noisy):+.1f}% with reward noise). Best QRC's error stays flat as the number of chains grows, "
                   "so the network already fills in rarely visited chains, the job expected traces do in the tabular case." if noisy else ""),
         pair(("Value error vs number of chains | σ=1", "m_x1", ["m_s1/best_qrc", "m_s1/qrc08", "m_s1/best_et"],
               {"m_s1/best_qrc": "best QRC", "m_s1/qrc08": "QRC λ=0.8", "m_s1/best_et": "best QRC-ET"},
               {"log_x": True, "title_x": "number of chains", "title_y": "value error (AUC)"}),
              ("Value error vs number of chains | σ=0", "m_x0", ["m_s0/best_qrc", "m_s0/qrc08", "m_s0/best_et"],
               {"m_s0/best_qrc": "best QRC", "m_s0/qrc08": "QRC λ=0.8", "m_s0/best_et": "best QRC-ET"},
               {"log_x": True, "title_x": "number of chains", "title_y": "value error (AUC)"})),
         pair(("m=32, σ=1 | value error during training", "mc_step", ["mc32/best_qrc", "mc32/qrc08", "mc32/best_et"],
               {"mc32/best_qrc": "best QRC", "mc32/qrc08": "QRC λ=0.8", "mc32/best_et": "best QRC-ET"},
               {"title_x": "Environment steps", "title_y": "value error", "range_y": (0, 0.3)}),
              ("m=1, σ=1 | value error during training", "mc_step", ["mc1/best_qrc", "mc1/qrc08", "mc1/best_et"],
               {"mc1/best_qrc": "best QRC", "mc1/qrc08": "QRC λ=0.8", "mc1/best_et": "best QRC-ET"},
               {"title_x": "Environment steps", "title_y": "value error", "range_y": (0, 0.3)})),
         MD("## Θ Target and Learning Rate (R31)"),
         MD("Training Θ toward the ordinary trace instead of Algorithm 3's recursive e_w target fits the trace better "
                   "and is slightly better in every task, but does not close the gap to QRC."),
         table("Θ Target and Learning Rate | m=32, σ=1", h=9),
         pair(("m=32, σ=1 | Θ trace loss (Θ lr 0.1)", "mc_step", ["tl/raw", "tl/e_w"],
               {"tl/raw": "ordinary-trace target", "tl/e_w": "Algorithm 3 e_w target"},
               {"title_x": "Environment steps", "title_y": "trace loss", "log_y": True}),
              ("m=128, σ=1 | value error during training", "mc_step", ["mc128/best_qrc", "mc128/qrc08", "mc128/best_et"],
               {"mc128/best_qrc": "best QRC", "mc128/qrc08": "QRC λ=0.8", "mc128/best_et": "best QRC-ET"},
               {"title_x": "Environment steps", "title_y": "value error", "range_y": (0, 0.3)})),
         MD("# Updated Decision"),
         MD("Once QRC's λ and learning rate are tuned, expected traces give no advantage in these tests: on Freeway the R27 gain "
                   "is explained by trace length and learning rate, and on the paper's own multi-chain task deep QRC-ET does not beat "
                   "tuned QRC. The ordinary-trace Θ target is the better of the two Θ targets."),
         MD("# Follow-up Reproduction and Evidence"),
         table("Reproduction | R28-R31", h=7),
         MD(f"Code and scripts: branch [cluster-r29]({GITHUB}). Every run is in this project (config experiment = R28…R31); "
            "exact code is in the artifact qrc-et-cluster-code-r27-r31.")]
    return B


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=REPORT_URL)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--tables", choices=["media", "panel", "markdown"], default="media")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="reuse an already logged FOLLOW-UP data run")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    import wandb_workspaces.reports.v2 as wr
    args.url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]

    if args.inspect:
        rep = wr.Report.from_url(args.url)
        print("title:", rep.title)
        for b in rep.blocks:
            name = type(b).__name__
            extra = getattr(b, "text", "") if name.startswith("H") else ""
            if name == "MarkdownBlock":
                extra = b.text[:90].replace("\n", " / ")
            if name == "PanelGrid":
                extra = [f"{type(p).__name__}:{getattr(p, 'title', '')} keys={getattr(p, 'media_keys', getattr(p, 'y', ''))} "
                         f"layout=({p.layout.x},{p.layout.y},{p.layout.w},{p.layout.h})" for p in b.panels] + \
                        [f"runset '{r.name}' filters={r.filters!r}" for r in b.runsets]
            print(f"  {name}: {extra}")
        return

    runs = load_runs()
    print(f"loaded {len(runs)} complete runs")
    D = build_data(runs)
    for name, (cols, rows) in D["tables"].items():
        print(f"\n{name}\n  " + " | ".join(cols))
        for r in rows:
            print("  " + " | ".join(str(c) for c in r))
    blocks = build_blocks(wr, D, args.entity, args.project, args.tables)
    print(f"\n{len(blocks)} blocks to append")
    if args.dry_run:
        return
    if not args.no_log:
        log_data_run(D, args.entity, args.project)
    if not args.no_backup:
        backup = wr.Report.from_url(args.url)
        backup.title = backup.title + " (backup before R28-R31 append)"
        backup.save(clone=True)
        print("Backup of the original report:", backup.url)
    rep = wr.Report.from_url(args.url)
    rep.blocks = list(rep.blocks) + blocks
    rep.width = "fluid"
    rep.save()
    print("Updated report:", rep.url)


if __name__ == "__main__":
    main()
