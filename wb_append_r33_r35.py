"""Append the R33-R35 follow-up (one-hot multi-chain, conv multi-chain, MinAtar with the fixes) to the EXISTING
report, in the same format as the R28-R31 sections: short paragraphs, one compact table panel per section,
then side-by-side comparison plots.

Steps it performs:
  1. Reads every R33, R34a, R34b, R34c, R35a, R35 run CSV and computes compact tables + mean curves.
  2. Logs them to one small online run "FOLLOW-UP | R33-R35 report data" (tables + plot series).
  3. Saves a BACKUP copy of the existing report, then appends the new sections to the original and saves it.

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_append_r33_r35.py --dry-run      # build and print the tables, log nothing, change nothing
    ~/wbenv/bin/python wb_append_r33_r35.py                # log data run, back up, append, save
"""
import argparse, os
from collections import defaultdict

import numpy as np

from wb_publish_runs import discover, read_run, run_summary, paired, mean_sem

REPORT_URL = ("https://forge.coreweave.com/mubbashirahmed/qrc-et-algo3-minatar/reports/"
              "QRC-vs-Experimental-QRC-ET-%7C-Five-Game-Screen-and-Breakout-Follow-up--VmlldzoxODExNTc2Nw==")
TAG = "FOLLOWUP-R33-R35"
GITHUB = "https://github.com/mubashir20023/eligibility-traces/tree/cluster-r29"
PREFIXES = ["R33", "R34a", "R34b", "R34c", "R35a", "R35"]
TASKS = ["m8-n4-p0.2", "m8-n4-p0.9", "m32-n4-p0.2", "m32-n4-p0.9"]
GAMES = ["Freeway", "Breakout", "Asterix"]
NICE = {
    "qrc_lam0": "QRC (λ=0)", "qrc_lam09": "QRC (λ=0.9)", "qrc_lam08": "QRC (λ=0.8)", "qrc_lam024": "QRC (λ=0.24)",
    "et_lag_default": "QRC-ET, current code (lag)", "et_lag_b01": "QRC-ET, lag + Θ step fixed",
    "et_nolag_b01": "QRC-ET, no lag + Θ step fixed", "et_nolag_b03": "QRC-ET, no lag + Θ step ×3",
    "et_split_b01": "QRC-ET, split (no lag + Θ step fixed)",
    "et_nolag_etlr01": "QRC-ET, no lag (Θ lr 0.1)", "et_nolag_etlr03": "QRC-ET, no lag (Θ lr 0.3)",
    "et_fix_eta0": "QRC-ET, no lag η=0", "et_fix_eta03": "QRC-ET, no lag η=0.3",
    "et_split_eta0": "QRC-ET, split η=0", "et_split_eta03": "QRC-ET, split η=0.3", "et_lag_eta03": "QRC-ET, current code η=0.3",
}


def nice(a):
    return NICE.get(a, a)


def task_name(t):
    m, _, p = t.split("-")
    return f"m={m[1:]}, p={p[1:]}"


# ------------------------------------------------------------------------------------------------ data
def load_runs():
    """{(prefix, env, agent, lr): {seed: run}} for complete runs."""
    out = defaultdict(dict)
    for p in discover(PREFIXES):
        d = os.path.dirname(p)
        parts = os.path.basename(d).split("__")
        exp, seed = parts[-2], parts[-1]
        pre, rest = exp.split("_", 1)
        if "_lr" not in rest or not seed.isdigit():
            continue
        agent, lr = rest.rsplit("_lr", 1)
        rel = os.path.relpath(d, "logs")
        if rel.startswith("multichain_env:"):
            env = rel.split("MultiChainSparse-")[1].split("-v0")[0]
        else:
            env = rel.split("/")[-1].split("-v1")[0]
        cols = read_run(p)
        total = 120_000 if pre in ("R33", "R34a", "R34b", "R35a") else 5_000_000
        s = run_summary(cols, total)
        if s.get("complete"):
            out[(pre, env, agent, lr)][int(seed)] = dict(cols=cols, summ=s)
    return out


def pooled(R, pres, env, agent, lr):
    d = {}
    for pre in pres:
        d.update(R.get((pre, env, agent, lr), {}))
    return d


def mean_curve(runs, key, grid, fill):
    ys = [np.interp(grid, r["cols"]["global_step"], np.nan_to_num(r["cols"][key], nan=fill, posinf=fill, neginf=fill))
          for r in runs]
    return np.mean(ys, 0).tolist() if ys else None


def best_setting(R, pres, env, agents, metric, lower):
    """Best (agent, lr) over the given agents by mean metric; returns (mean, agent, lr, {seed: value})."""
    cand = []
    keys = {(k[2], k[3]) for k in R if k[0] in pres and k[1] == env and k[2] in agents}
    for a, lr in keys:
        d = pooled(R, pres, env, a, lr)
        v = {s: r["summ"][metric] for s, r in d.items() if metric in r["summ"]}
        if len(v) >= 2:
            cand.append((np.mean(list(v.values())), a, lr, v))
    if not cand:
        return None
    return min(cand, key=lambda c: c[0] if lower else -c[0])


def pct_row(et, q, lower=True):
    c = paired(et[3], q[3])
    if not c:
        return "–", "–"
    wins = (c["n"] - c["a_higher"]) if lower else c["a_higher"]
    return f"{100 * (et[0] / q[0] - 1):+.1f}%", f"{wins}/{c['n']}"


def build_data(R):
    D = {"tables": {}, "series": {}}
    AUC = "value_rmse_auc"
    QRC_MC = ["qrc_lam0", "qrc_lam09"]

    # --- R33: which fix matters (10 seeds)
    rows = []
    for t in TASKS:
        q = best_setting(R, ["R33"], t, QRC_MC, AUC, True)
        if not q:
            continue
        row = [task_name(t), f"{q[0]:.4f}", f"{nice(q[1])}, lr {q[2]}"]
        for a in ["et_lag_default", "et_lag_b01", "et_nolag_b03", "et_nolag_b01"]:
            e = best_setting(R, ["R33"], t, [a], AUC, True)
            row.append(pct_row(e, q)[0] if e else "–")
        rows.append(row)
    D["tables"]["One-Hot Multi-Chain | Which Fix Matters (R33)"] = (
        ["Task", "Best QRC error", "Best QRC setting", "Current code (lag)", "Lag + Θ step fixed",
         "No lag + Θ step ×3", "No lag + Θ step fixed"], rows)

    # --- R34a + R35a: fresh seeds 11-30
    rows = []
    for t in TASKS:
        q = best_setting(R, ["R34a"], t, QRC_MC, AUC, True)
        if not q:
            continue
        row = [task_name(t), f"{q[0]:.4f}", f"{nice(q[1])}, lr {q[2]}"]
        for a, pres in [("et_nolag_b01", ["R34a"]), ("et_split_b01", ["R35a"]), ("et_lag_default", ["R34a"])]:
            e = best_setting(R, pres, t, [a], AUC, True)
            if e:
                pc, w = pct_row(e, q)
                row += [f"{e[0]:.4f}", f"{pc} ({w})"]
            else:
                row += ["–", "–"]
        rows.append(row)
    D["tables"]["One-Hot Multi-Chain | 20 Fresh Seeds (R34a, R35a)"] = (
        ["Task", "Best QRC error", "Best QRC setting", "No-lag ET error", "No-lag vs QRC (better on)",
         "Split ET error", "Split vs QRC (better on)", "Current code error", "Current code vs QRC (better on)"], rows)

    # --- R34b: conv network
    rows = []
    for t in TASKS:
        q = best_setting(R, ["R34b"], t, QRC_MC, AUC, True)
        q9 = best_setting(R, ["R34b"], t, ["qrc_lam09"], AUC, True)
        if not q:
            continue
        row = [task_name(t), f"{q[0]:.4f}", f"{nice(q[1])}, lr {q[2]}", f"{q9[0]:.4f}" if q9 else "–"]
        for a in ["et_lag_default", "et_nolag_etlr01", "et_nolag_etlr03"]:
            e = best_setting(R, ["R34b"], t, [a], AUC, True)
            row.append(pct_row(e, q)[0] if e else "–")
        rows.append(row)
    D["tables"]["Conv Network | Multi-Chain (R34b)"] = (
        ["Task", "Best QRC error", "Best QRC setting", "QRC λ=0.9 error", "Current code (lag)",
         "No lag, Θ lr 0.1", "No lag, Θ lr 0.3"], rows)

    # --- MinAtar: R34c + R35
    MP = ["R34c", "R35"]
    rows = []
    for g in GAMES:
        q = best_setting(R, MP, g, ["qrc_lam024", "qrc_lam08"], "final_score", False)
        if not q:
            continue
        for a in ["qrc_lam024", "qrc_lam08", "et_split_eta03", "et_split_eta0", "et_lag_eta03", "et_fix_eta03", "et_fix_eta0"]:
            e = best_setting(R, MP, g, [a], "final_score", False)
            if not e:
                continue
            runs = pooled(R, MP, g, a, e[2])
            div = sum(bool(r["summ"].get("diverged")) for r in runs.values())
            mu, se = mean_sem(e[3].values())
            c = paired(e[3], q[3]) if a.startswith("et") else None
            rows.append([g, nice(a), e[2], f"{mu:.1f} ± {se:.1f}", f"{div}/{len(runs)}",
                         f"{c['diff']:+.1f} [{c['ci_low']:+.1f}, {c['ci_high']:+.1f}]" if c else "–"])
    D["tables"]["MinAtar | Final Score with the Fixes (R34c, R35)"] = (
        ["Game", "Agent", "Best lr", "Score", "Diverged", "ET − best QRC (95% CI)"], rows)

    # --- series: one-hot value error curves (20 seeds)
    gx = np.arange(1000, 120_001, 1000)
    for t, tag in [("m8-n4-p0.2", "oh8"), ("m32-n4-p0.9", "oh32")]:
        q = best_setting(R, ["R34a"], t, QRC_MC, AUC, True)
        S = {"x": gx.tolist()}
        if q:
            S["qrc"] = mean_curve(pooled(R, ["R34a"], t, q[1], q[2]).values(), "value_rmse", gx, 10.0)
        for a, pres, key in [("et_split_b01", ["R35a"], "split"), ("et_lag_default", ["R34a"], "lag")]:
            e = best_setting(R, pres, t, [a], AUC, True)
            if e:
                S[key] = mean_curve(pooled(R, pres, t, a, e[2]).values(), "value_rmse", gx, 10.0)
        D["series"][tag] = S
    # conv
    t = "m32-n4-p0.9"
    S = {"x": gx.tolist()}
    for agents, key in [(["qrc_lam0"], "qrc0"), (["qrc_lam09"], "qrc09"), (["et_nolag_etlr01"], "nolag"), (["et_lag_default"], "lag")]:
        e = best_setting(R, ["R34b"], t, agents, AUC, True)
        if e:
            S[key] = mean_curve(pooled(R, ["R34b"], t, e[1], e[2]).values(), "value_rmse", gx, 10.0)
    D["series"]["conv32"] = S

    # MinAtar curves
    g25 = np.arange(25_000, 5_000_001, 25_000)
    for g in GAMES:
        S = {"x": g25.tolist()}
        q = best_setting(R, MP, g, ["qrc_lam024", "qrc_lam08"], "final_score", False)
        if q:
            S["qrc"] = mean_curve(pooled(R, MP, g, q[1], q[2]).values(), "avg_return", g25, 0.0)
        for a, key in [("et_split_eta03", "split03"), ("et_fix_eta03", "nolag03")]:
            e = best_setting(R, MP, g, [a], "final_score", False)
            if e:
                S[key] = mean_curve(pooled(R, MP, g, a, e[2]).values(), "avg_return", g25, 0.0)
        D["series"]["mn_" + g.lower()] = S
    # Freeway eta=0 seeds (collapse)
    S = {"x": g25.tolist()}
    for s, r in sorted(pooled(R, ["R35"], "Freeway", "et_split_eta0", "2e-3").items()):
        S[f"s{s}"] = mean_curve([r], "avg_return", g25, 0.0)
    D["series"]["fw_eta0"] = S

    D["tables"]["Reproduction | R33-R35"] = (["Experiment", "SLURM job", "Job script", "Runs"], [
        ["R33 One-hot screen", "19489436", "slurm/R33_onehot_sparse.sh", 720],
        ["R34a One-hot, fresh seeds", "–", "slurm/R34a_onehot_confirm.sh", 420],
        ["R34b Conv network, multi-chain", "–", "slurm/R34b_conv_fixed.sh", 600],
        ["R34c MinAtar, no lag", "19497492", "slurm/R34c_minatar_fixed.sh", 120],
        ["R35a One-hot, split", "19569543", "slurm/R35a_onehot_split.sh", 80],
        ["R35 MinAtar, split", "19569571", "slurm/R35_minatar_split.sh", 90]])
    return D


# ------------------------------------------------------------------------------------------------ data run
STEPS = [("oh8", "oh_step"), ("oh32", "oh_step"), ("conv32", "oh_step"),
         ("mn_freeway", "mn_step"), ("mn_breakout", "mn_step"), ("mn_asterix", "mn_step"), ("fw_eta0", "mn_step")]


def log_data_run(D, entity, project):
    import wandb
    run = wandb.init(entity=entity, project=project, name="FOLLOW-UP | R33-R35 report data", job_type="report-data",
                     group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
    for prefix, step in STEPS:
        wandb.define_metric(step)
        wandb.define_metric(f"{prefix}/*", step_metric=step)
    for prefix, step in STEPS:
        s = D["series"].get(prefix)
        if not s:
            continue
        for j, x in enumerate(s["x"]):
            row = {step: x}
            for k, v in s.items():
                if k != "x" and v is not None and v[j] is not None:
                    row[f"{prefix}/{k}"] = v[j]
            wandb.log(row)
    for name, (cols, rows) in D["tables"].items():
        wandb.log({name: wandb.Table(columns=cols, data=[[str(c) for c in r] for r in rows])})
    run.finish()


# ------------------------------------------------------------------------------------------------ report blocks
def build_blocks(wr, D, entity, project):
    rs = wr.Runset(entity=entity, project=project, name="Report tables | R33-R35", filters=f"Config('experiment') == '{TAG}'")
    T = D["tables"]

    def table(name):
        h = 3 + len(T[name][1])
        return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title=name, media_keys=[name], num_columns=1,
                                                                  layout=wr.Layout(x=0, y=0, w=24, h=h))])

    def pair(left, right, h=9):
        panels = []
        for i, p in enumerate([left, right]):
            if p:
                title, x, ys, titles, extra = p
                panels.append(wr.LinePlot(title=title, x=x, y=ys, line_titles=titles, layout=wr.Layout(x=12 * i, y=0, w=12, h=h), **extra))
        return wr.PanelGrid(runsets=[rs], panels=panels)

    MD = lambda text: wr.MarkdownBlock(text=text)
    ve = {"title_x": "Environment steps", "title_y": "value error over all states", "range_y": (0, 0.15)}
    ret = {"title_x": "Environment steps", "title_y": "Rolling-100-episode training return"}
    oh = {"oh8/qrc": "best QRC", "oh8/split": "QRC-ET, split", "oh8/lag": "QRC-ET, current code (lag)"}
    oh32 = {k.replace("oh8", "oh32"): v for k, v in oh.items()}
    fw0 = [k for k in D["series"].get("fw_eta0", {}) if k != "x"]
    return [
        wr.HorizontalRule(),
        MD("# Follow-up | Expected Traces with One-Hot Features (R33-R35a)"),
        MD("Peter's suggestion: remove the conv layers, so each state is a one-hot input and the deep agent becomes tabular. "
           "Mathieu's suggestion: sparse reward (+1 with probability p at the shared state). Metric: value error over all "
           "states (lower is better). Two problems were found in the expected-trace agent: (1) Algorithm 3 applies Θ's "
           "prediction one step late (the lag), and (2) Θ's step size was about 100× too small, so z stayed near its random start."),
        table("One-Hot Multi-Chain | Which Fix Matters (R33)"),
        MD("Percentages are QRC-ET's value error relative to the best tuned QRC (any λ, any learning rate), 10 seeds. "
           "Both fixes are needed: fixing only Θ's step size still loses by 44-80%."),
        MD("## One-Hot | 20 Fresh Seeds and the Split Form"),
        MD("Seeds 11-30, never used before, each agent at the learning rate R33 picked. The split form (Θ predicts only "
           "the past part of the trace; the current gradient stays exact) is equal or better than plain no-lag."),
        table("One-Hot Multi-Chain | 20 Fresh Seeds (R34a, R35a)"),
        pair(("m=8, p=0.2 | value error during training", "oh_step", list(oh), oh, ve),
             ("m=32, p=0.9 | value error during training", "oh_step", list(oh32), oh32, ve)),
        MD("# Follow-up | Conv Network on the Same Task (R34b)"),
        MD("Same sparse tasks with the usual MinAtar conv network. Here the best QRC uses no trace at all (λ=0); "
           "every QRC-ET version is worse than it, but better than QRC with λ=0.9. The conv features collapse "
           "(trace diversity → 0, effective rank 61 → 3-8), so the network already shares credit across chains and the "
           "expected trace has nothing to average."),
        table("Conv Network | Multi-Chain (R34b)"),
        pair(("Conv, m=32, p=0.9 | value error during training", "oh_step", ["conv32/qrc0", "conv32/qrc09", "conv32/nolag", "conv32/lag"],
              {"conv32/qrc0": "QRC (λ=0)", "conv32/qrc09": "QRC (λ=0.9)", "conv32/nolag": "QRC-ET, no lag", "conv32/lag": "QRC-ET, current code"},
              ve), None),
        MD("# Follow-up | MinAtar with the Fixes (R34c, R35)"),
        MD("5 seeds (201-205), 5M steps, learning rates 5e-4 and 2e-3. Removing the lag alone makes QRC-ET diverge on "
           "Freeway and Asterix. The split form never diverged; with η=0.3 it ties tuned QRC on Freeway and Breakout and "
           "is below the best QRC (λ=0.24) on Asterix. ET was only run with λ=0.8; at matched λ=0.8 it is level with QRC "
           "on all three games."),
        table("MinAtar | Final Score with the Fixes (R34c, R35)"),
        pair(("Freeway | best QRC vs QRC-ET", "mn_step", ["mn_freeway/qrc", "mn_freeway/split03", "mn_freeway/nolag03"],
              {"mn_freeway/qrc": "best QRC", "mn_freeway/split03": "QRC-ET, split η=0.3", "mn_freeway/nolag03": "QRC-ET, no lag η=0.3"}, ret),
             ("Asterix | best QRC vs QRC-ET", "mn_step", ["mn_asterix/qrc", "mn_asterix/split03", "mn_asterix/nolag03"],
              {"mn_asterix/qrc": "best QRC", "mn_asterix/split03": "QRC-ET, split η=0.3", "mn_asterix/nolag03": "QRC-ET, no lag η=0.3"}, ret)),
        pair(("Breakout | best QRC vs QRC-ET", "mn_step", ["mn_breakout/qrc", "mn_breakout/split03", "mn_breakout/nolag03"],
              {"mn_breakout/qrc": "best QRC", "mn_breakout/split03": "QRC-ET, split η=0.3", "mn_breakout/nolag03": "QRC-ET, no lag η=0.3"}, ret),
             ("Freeway | split η=0, each seed (lr 2e-3)", "mn_step", [f"fw_eta0/{k}" for k in fw0],
              {f"fw_eta0/{k}": f"seed {k[1:]}" for k in fw0}, ret) if fw0 else None),
        MD("With η=0 (pure expected trace), 4 of 10 Freeway runs learned and then collapsed to return 0 (Q → 0), "
           "rather than diverging. Keeping 30% of the ordinary trace (η=0.3) avoids it at lr 5e-4."),
        MD("# Updated Decision | R33-R35"),
        MD("1. With features that keep states apart, compiled expected traces beat tuned QRC by 13-32% on every task and "
           "every one of 20 fresh seeds.\n"
           "2. This needs the one-step lag removed and a correctly sized Θ step; the split form makes it stable with deep networks.\n"
           "3. With conv features, QRC-ET is stable and level with tuned QRC on MinAtar but does not beat it: the representation "
           "collapses states, which limits what expected traces can add."),
        MD("# Follow-up Reproduction | R33-R35"),
        table("Reproduction | R33-R35"),
        MD(f"Code and scripts: branch [cluster-r29]({GITHUB}), see RESULTS_R33_R35.md. Every run is in this project "
           "(config experiment = R33 … R35)."),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=REPORT_URL)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="reuse an already logged FOLLOW-UP R33-R35 data run")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()
    args.url = args.url.replace("forge.coreweave.com/wandb/", "forge.coreweave.com/").split("?")[0]

    R = load_runs()
    n = defaultdict(int)
    for k, v in R.items():
        n[k[0]] += len(v)
    print("complete runs:", dict(n))
    D = build_data(R)
    for name, (cols, rows) in D["tables"].items():
        print(f"\n{name}\n  " + " | ".join(cols))
        for r in rows:
            print("  " + " | ".join(str(c) for c in r))
    import wandb_workspaces.reports.v2 as wr
    blocks = build_blocks(wr, D, args.entity, args.project)
    print(f"\n{len(blocks)} blocks to append")
    if args.dry_run:
        return
    if not args.no_log:
        log_data_run(D, args.entity, args.project)
    if not args.no_backup:
        backup = wr.Report.from_url(args.url)
        backup.title = backup.title + " (backup before R33-R35 append)"
        backup.save(clone=True)
        print("Backup of the original report:", backup.url)
    rep = wr.Report.from_url(args.url)
    rep.blocks = list(rep.blocks) + blocks
    rep.width = "fluid"
    rep.save()
    print("Updated report:", rep.url)


if __name__ == "__main__":
    main()
