"""Publish every QRC / QRC-ET cluster run (R27-R31 by default) to W&B, plus one analysis run with all
summary tables and figures used by the report.

What it does
  1. Finds every logs/**/<env>__<exp_name>__<seed>/training_data.csv whose exp_name starts with one of
     the prefixes (default R27_ ... R31_), reads the run's own config.json, and creates one W&B run with:
       - config   = the run's full config.json + experiment / arm / env / multi-chain parameters / source path
       - history  = every numeric CSV column (Freeway runs thinned to one row per 5k steps; scores use all rows)
       - summary  = final score, curve average, divergence flag, value-error AUC/final, trace loss, row coverage
  2. Computes the R28-R31 analysis (the same numbers as summarize_r28.py / summarize_sweep.py /
     summarize_r30.py / summarize_r31.py) and writes report_data.json + figures/*.png.
  3. Logs one extra run "ANALYSIS | R27-R31 tables and figures" with every table (wandb.Table) and figure.

Runs are written OFFLINE (no network needed, so it can run in a batch job); upload them afterwards with
    cd <out> && wandb sync --sync-all
A ledger (<out>/published.json) records what has been written, so re-running skips finished runs.

Usage (from the stream-rep-rl folder):
    python wb_publish_runs.py --entity mubbashirahmed --project qrc-et-algo3-minatar
    python wb_publish_runs.py --prefixes R12 R13 R14 R15 R26 R27   # older experiments, if their logs exist
    python wb_publish_runs.py --analysis-only                      # just rebuild tables/figures
"""
import argparse, csv, glob, hashlib, json, math, os, re, sys
from collections import defaultdict

import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

REPO = os.path.abspath(".")
GITHUB = "https://github.com/mubashir20023/eligibility-traces/tree/cluster-r29"
SLURM_JOBS = {"R27": "18568147", "R28": "19078432", "R29": "19146242",
              "R30": "19288335 (duplicate 19288347 cancelled before any task started)", "R31": "19290244",
              "R33": "19489436", "R34c": "19497492", "R35a": "19569543", "R35": "19569571"}
SCRIPTS = {"R27": "slurm/R27_freeway_2e3.sh", "R28": "slurm/R28_freeway_controls.sh",
           "R29": "slurm/R29_freeway_lr_sweep.sh", "R30": "slurm/R30_pred_screen.sh", "R31": "slurm/R31_theta_fix.sh",
           "R33": "slurm/R33_onehot_sparse.sh", "R34a": "slurm/R34a_onehot_confirm.sh", "R34b": "slurm/R34b_conv_fixed.sh",
           "R34c": "slurm/R34c_minatar_fixed.sh", "R35a": "slurm/R35a_onehot_split.sh", "R35": "slurm/R35_minatar_split.sh"}


# ----------------------------------------------------------------------------------------------- reading
def read_run(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    cols = {}
    for k in (rows[0].keys() if rows else []):
        try:
            cols[k] = np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in rows])
        except ValueError:
            pass                                             # non-numeric column
    return cols


def run_info(csv_path):
    run_dir = os.path.dirname(csv_path)
    base = os.path.basename(run_dir)
    parts = base.split("__")
    if len(parts) < 3:
        return None
    exp, seed = parts[-2], parts[-1]
    cfg = {}
    if os.path.exists(os.path.join(run_dir, "config.json")):
        with open(os.path.join(run_dir, "config.json")) as f:
            cfg = json.load(f)
    env_id = cfg.get("env_id") or os.path.relpath(run_dir, "logs").rsplit("__", 2)[0]
    experiment = exp.split("_")[0]
    is_et = any(k.startswith("et_") for k in cfg) or "_et_" in exp or "algo3" in exp
    lam, eta = cfg.get("lamda"), cfg.get("et_eta")
    if is_et:
        label = f"ET eta={eta}"
        if cfg.get("et_zero_theta"):
            label += " Theta-off"
        if cfg.get("et_algo3_target") not in (None, "e_w"):
            label += f" target={cfg['et_algo3_target']}"
        if cfg.get("et_lr") not in (None, 0.1):
            label += f" Theta-lr={cfg['et_lr']}"
        if cfg.get("et_algo3_lag") is False:
            label += " split" if cfg.get("et_split") else " no-lag"
    else:
        label = f"QRC lambda={lam}"
    if cfg.get("net_arch") == "linear":
        label += " one-hot"
    info = dict(experiment=experiment, exp_name=exp, seed=int(seed) if seed.isdigit() else seed, env_id=env_id,
                agent="ET" if is_et else "QRC", arm=label, q_lr=cfg.get("q_lr"), lamda=lam, et_eta=eta,
                run_dir=os.path.abspath(run_dir), slurm_job=SLURM_JOBS.get(experiment, ""),
                slurm_script=SCRIPTS.get(experiment, ""), code=GITHUB)
    m = re.search(r"MultiChainPred-m(\d+)-n(\d+)-s([\d.]+)", env_id)
    if m:
        info.update(chains=int(m.group(1)), chain_len=int(m.group(2)), reward_noise=float(m.group(3)))
    m = re.search(r"MultiChainSparse-m(\d+)-n(\d+)-p([\d.]+)", env_id)
    if m:
        info.update(chains=int(m.group(1)), chain_len=int(m.group(2)), reward_prob=float(m.group(3)))
    return info, cfg


def run_summary(cols, total_steps):
    s = {}
    step = cols.get("global_step", np.array([]))
    if len(step) == 0:
        return {"rows": 0}
    last = step > 0.9 * step.max()
    s.update(rows=int(len(step)), first_logged_step=int(step.min()), last_logged_step=int(step.max()),
             complete=bool(step.max() >= 0.99 * total_steps) if total_steps else None,
             starts_at_beginning=bool(step.min() <= 1000))
    if "avg_return" in cols:
        r = cols["avg_return"]
        s.update(final_score=float(np.nanmean(r[last])), curve_avg=float(np.nanmean(r)))
    if "q_values" in cols:
        q = cols["q_values"]
        s["diverged"] = bool(np.any(~np.isfinite(q)) or np.nanmax(np.abs(q)) > 1e4)
    if "value_rmse" in cols:
        v = np.where(np.isfinite(cols["value_rmse"]), cols["value_rmse"], 10.0)
        s.update(value_rmse_auc=float(v.mean()), value_rmse_final=float(v[last].mean()))
    if "et_trace_loss" in cols:
        s["trace_loss_final"] = float(np.nanmean(cols["et_trace_loss"][last]))
    return s


def discover(prefixes):
    found = []
    for p in sorted(glob.glob("logs/**/training_data.csv", recursive=True)):
        parts = os.path.basename(os.path.dirname(p)).split("__")
        if len(parts) >= 3 and any(parts[-2].startswith(pre + "_") for pre in prefixes):
            found.append(p)
    return found


# ----------------------------------------------------------------------------------------------- upload
def publish(paths, args):
    import wandb
    os.makedirs(args.out, exist_ok=True)
    ledger_path = os.path.join(args.out, "published.json")
    ledger = json.load(open(ledger_path)) if os.path.exists(ledger_path) else {}
    for i, p in enumerate(paths):
        key = os.path.abspath(p)
        if key in ledger:
            continue
        parsed = run_info(p)
        if parsed is None:
            continue
        info, cfg = parsed
        cols = read_run(p)
        total = cfg.get("total_timesteps")
        summ = run_summary(cols, total)
        step = cols.get("global_step", np.array([]))
        keep = np.ones(len(step), bool)
        if len(step) > 2000:                                  # thin long Freeway histories for plotting
            keep = (step.astype(int) % args.stride == 0) | (np.arange(len(step)) == len(step) - 1)
        salt = "" if os.path.basename(os.path.normpath(args.out)) == "wandb_upload" else os.path.basename(os.path.normpath(args.out))
        run_id = hashlib.md5((key + salt).encode()).hexdigest()[:12]   # new folder name -> new run ids
        run = wandb.init(mode="offline", dir=args.out, entity=args.entity, project=args.project, id=run_id,
                         name=f"{info['experiment']} | {info['env_id'].split(':')[-1]} | {info['arm']} | lr {info['q_lr']} | seed {info['seed']}",
                         group=info["experiment"], job_type=info["arm"],
                         tags=[info["experiment"], info["agent"], "cluster-import"],
                         config={**cfg, **info, "imported_from": key}, reinit=True)
        for j in np.where(keep)[0]:
            run.log({k: (float(v[j]) if np.isfinite(v[j]) else None) for k, v in cols.items()})
        run.summary.update({k: v for k, v in summ.items() if v is not None})
        run.finish()
        ledger[key] = run_id
        if (i + 1) % 25 == 0 or i + 1 == len(paths):
            json.dump(ledger, open(ledger_path, "w"), indent=1)
            print(f"  {i + 1}/{len(paths)} written", flush=True)
    json.dump(ledger, open(ledger_path, "w"), indent=1)


# ----------------------------------------------------------------------------------------------- analysis
def paired(a, b):
    """a, b: {seed: value}. Returns dict with n, diff, ci, p, wins (a > b)."""
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return None
    d = np.array([a[s] - b[s] for s in common])
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1) if stats else 2.145
    p = float(stats.ttest_rel([a[s] for s in common], [b[s] for s in common]).pvalue) if stats else float("nan")
    return dict(n=len(d), diff=float(d.mean()), ci_low=float(d.mean() - t * se), ci_high=float(d.mean() + t * se),
                p=p, a_higher=int((d > 0).sum()))


def mean_sem(v):
    v = np.array(list(v))
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def analyse(out):
    runs = []
    for p in discover(["R27", "R28", "R29", "R30", "R31"]):
        parsed = run_info(p)
        if parsed is None:
            continue
        info, cfg = parsed
        cols = read_run(p)
        runs.append(dict(info=info, cfg=cfg, cols=cols, summ=run_summary(cols, cfg.get("total_timesteps"))))
    data = {"coverage": {}, "tables": {}}
    for e in ["R27", "R28", "R29", "R30", "R31"]:
        rs = [r for r in runs if r["info"]["experiment"] == e]
        data["coverage"][e] = dict(runs=len(rs), complete=sum(bool(r["summ"].get("complete")) for r in rs),
                                   start_at_step_1000=sum(bool(r["summ"].get("starts_at_beginning")) for r in rs),
                                   diverged=sum(bool(r["summ"].get("diverged")) for r in rs))

    # ---------- Freeway (R28 + R29): final score per (arm, lr)
    fw = defaultdict(dict)
    for r in runs:
        i = r["info"]
        if i["experiment"] in ("R28", "R29") and r["summ"].get("complete"):
            fw[(i["arm"], i["q_lr"])][i["seed"]] = r["summ"]["final_score"]
    grid = []
    for (arm, lr), d in sorted(fw.items(), key=lambda x: (x[0][0], x[0][1])):
        mu, se = mean_sem(d.values())
        div = sum(1 for r in runs if r["info"]["experiment"] in ("R28", "R29") and r["info"]["arm"] == arm
                  and r["info"]["q_lr"] == lr and r["summ"].get("diverged"))
        grid.append(dict(arm=arm, q_lr=lr, n=len(d), mean=mu, sem=se, diverged=div))
    data["tables"]["freeway_grid"] = grid
    ref = fw.get(("ET eta=0.3", 0.002), {})
    comps = []
    for (arm, lr), d in sorted(fw.items()):
        if (arm, lr) != ("ET eta=0.3", 0.002):
            c = paired(ref, d)
            if c:
                comps.append(dict(versus=f"{arm} @ lr {lr}", **c))
    data["tables"]["freeway_paired_vs_et_lr2e-3"] = comps

    # ---------- Multi-chain (R30 + R31): value-error AUC per (env, arm, lr)
    mc = defaultdict(dict)
    tl = defaultdict(dict)
    for r in runs:
        i = r["info"]
        if i["experiment"] in ("R30", "R31") and r["summ"].get("complete") and "value_rmse_auc" in r["summ"]:
            k = (f"m{i['chains']}-s{i['reward_noise']}", i["experiment"], i["arm"], i["q_lr"])
            mc[k][i["seed"]] = r["summ"]["value_rmse_auc"]
            if "trace_loss_final" in r["summ"]:
                tl[k][i["seed"]] = r["summ"]["trace_loss_final"]
    rows = []
    for k, d in sorted(mc.items()):
        mu, se = mean_sem(d.values())
        rows.append(dict(env=k[0], experiment=k[1], arm=k[2], q_lr=k[3], n=len(d), auc=mu, sem=se,
                         trace_loss=(mean_sem(tl[k].values())[0] if tl.get(k) else None)))
    data["tables"]["multichain_all"] = rows
    best = []
    for env in sorted({k[0] for k in mc}, key=lambda e: (float(e.split("-s")[1]), int(e[1:].split("-")[0]))):
        q = [(np.mean(list(d.values())), k) for k, d in mc.items() if k[0] == env and k[2].startswith("QRC") and len(d) >= 5]
        e_ = [(np.mean(list(d.values())), k) for k, d in mc.items() if k[0] == env and k[2].startswith("ET") and len(d) >= 5]
        if not q or not e_:
            continue
        qa, qk = min(q)
        ea, ek = min(e_)
        c = paired(mc[ek], mc[qk])
        best.append(dict(env=env, best_qrc=f"{qk[2]} lr {qk[3]}", best_qrc_auc=float(qa),
                         best_et=f"{ek[1]} {ek[2]} lr {ek[3]}", best_et_auc=float(ea),
                         et_minus_qrc_pct=float(100 * (ea / qa - 1)),
                         et_lower_on=(c["n"] - c["a_higher"]) if c else None, n=c["n"] if c else None))
    data["tables"]["multichain_best"] = best

    with open(os.path.join(out, "report_data.json"), "w") as f:
        json.dump(data, f, indent=1, default=float)
    figs = make_figures(runs, data, fw, mc, os.path.join(out, "figures"))
    return data, figs


def make_figures(runs, data, fw, mc, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)
    figs = {}
    colors = {"QRC lambda=0.8": "#1f77b4", "QRC lambda=0.24": "#2ca02c", "QRC lambda=0.0": "#9467bd",
              "ET eta=0.3": "#d62728", "ET eta=0.0": "#ff7f0e", "ET eta=0.3 Theta-off": "#7f7f7f"}

    # 1. Freeway learning-rate sweep
    if data["tables"]["freeway_grid"]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for arm in sorted({g["arm"] for g in data["tables"]["freeway_grid"]}):
            g = sorted([x for x in data["tables"]["freeway_grid"] if x["arm"] == arm], key=lambda x: x["q_lr"])
            ax.errorbar([x["q_lr"] for x in g], [x["mean"] for x in g], yerr=[x["sem"] for x in g], marker="o",
                        capsize=3, label=arm, color=colors.get(arm))
        ax.set_xscale("log"); ax.set_xlabel("Q learning rate"); ax.set_ylabel("Final score (last 10% of 5M steps)")
        ax.set_title("Freeway: final score vs learning rate (15 seeds, mean ± s.e.)"); ax.legend(fontsize=8)
        ax.grid(alpha=.3); fig.tight_layout()
        figs["fig/freeway_lr_sweep"] = os.path.join(figdir, "freeway_lr_sweep.png"); fig.savefig(figs["fig/freeway_lr_sweep"], dpi=150); plt.close(fig)

    # 2. Freeway learning curves for the R28 arms
    curves = defaultdict(list)
    for r in runs:
        i = r["info"]
        if i["experiment"] == "R28" and r["summ"].get("complete"):
            curves[f"{i['arm']} lr {i['q_lr']}"].append((r["cols"]["global_step"], r["cols"]["avg_return"]))
    if curves:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        grid_x = np.arange(5000, 5_000_001, 5000)
        for name, cs in sorted(curves.items()):
            ys = np.array([np.interp(grid_x, s, np.nan_to_num(v)) for s, v in cs])
            mu, se = ys.mean(0), ys.std(0, ddof=1) / np.sqrt(len(ys))
            c = colors.get(name.split(" lr ")[0])
            ax.plot(grid_x, mu, label=f"{name} (n={len(ys)})", color=c, lw=1.2)
            ax.fill_between(grid_x, mu - se, mu + se, alpha=.2, color=c)
        ax.set_xlabel("Environment steps"); ax.set_ylabel("Rolling-100-episode training return")
        ax.set_title("Freeway R28: learning curves (mean ± s.e.)"); ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout()
        figs["fig/freeway_r28_curves"] = os.path.join(figdir, "freeway_r28_curves.png"); fig.savefig(figs["fig/freeway_r28_curves"], dpi=150); plt.close(fig)

    # 3. Multi-chain: best value error vs number of chains
    if mc:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, sig in zip(axes, ["0.0", "1.0"]):
            for fam, sel in [("best QRC (any lambda, lr)", lambda k: k[2].startswith("QRC")),
                             ("QRC lambda=0.8 (best lr)", lambda k: k[2] == "QRC lambda=0.8"),
                             ("best ET (R30 + R31)", lambda k: k[2].startswith("ET"))]:
                xs, ys, es = [], [], []
                for m in [1, 8, 32, 128]:
                    cand = [(np.mean(list(d.values())), d) for k, d in mc.items()
                            if k[0] == f"m{m}-s{sig}" and sel(k) and len(d) >= 5]
                    if cand:
                        mu, d = min(cand, key=lambda x: x[0])
                        xs.append(m); ys.append(mu); es.append(mean_sem(d.values())[1])
                if xs:
                    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=fam)
            ax.set_xscale("log", base=2); ax.set_xlabel("number of chains m"); ax.set_ylabel("value-error AUC (lower = better)")
            ax.set_title(f"Multi-chain prediction, reward noise sigma={sig}"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout()
        figs["fig/multichain_vs_chains"] = os.path.join(figdir, "multichain_vs_chains.png"); fig.savefig(figs["fig/multichain_vs_chains"], dpi=150); plt.close(fig)

    # 4. Multi-chain value-error curves at m32, sigma 1 (best setting per arm)
    sel = {}
    for k, d in mc.items():
        if k[0] == "m32-s1.0" and len(d) >= 5:
            key = f"{k[1]} {k[2]}"
            if key not in sel or np.mean(list(d.values())) < sel[key][0]:
                sel[key] = (np.mean(list(d.values())), k)
    if sel:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key, (_, k) in sorted(sel.items()):
            cs = [(r["cols"]["global_step"], r["cols"]["value_rmse"]) for r in runs
                  if r["info"]["experiment"] == k[1] and r["info"]["arm"] == k[2] and r["info"]["q_lr"] == k[3]
                  and r["info"].get("chains") == 32 and r["info"].get("reward_noise") == 1.0 and r["summ"].get("complete")]
            gx = np.arange(1000, 100_001, 1000)
            ys = np.array([np.interp(gx, s, np.nan_to_num(v, nan=10.0)) for s, v in cs])
            mu = ys.mean(0)
            ax.plot(gx, mu, lw=1, label=f"{key} lr {k[3]} (AUC {mu.mean():.4f})")
        ax.set_ylim(0, 0.4); ax.set_xlabel("Environment steps"); ax.set_ylabel("value RMSE over all states")
        ax.set_title("Multi-chain m=32, sigma=1: best setting per arm (5-seed mean)"); ax.legend(fontsize=6); ax.grid(alpha=.3); fig.tight_layout()
        figs["fig/multichain_m32_s1_curves"] = os.path.join(figdir, "multichain_m32_s1_curves.png"); fig.savefig(figs["fig/multichain_m32_s1_curves"], dpi=150); plt.close(fig)
    return figs


def log_analysis(args, data, figs):
    import wandb
    run = wandb.init(mode="offline", dir=args.out, entity=args.entity, project=args.project,
                     id="analysis-r27-r31", name="ANALYSIS | R27-R31 tables and figures", group="ANALYSIS",
                     job_type="analysis", config={"experiment": "ANALYSIS", "code": GITHUB}, reinit=True)
    for name, rows in data["tables"].items():
        if rows:
            cols = list(rows[0].keys())
            run.log({f"table/{name}": wandb.Table(columns=cols, data=[[r.get(c) for c in cols] for r in rows])})
    for key, path in figs.items():
        run.log({key: wandb.Image(path)})
    run.summary["coverage"] = data["coverage"]
    code = wandb.Artifact("qrc-et-cluster-code-r27-r31", type="code",
                          description="Agents, environment, patch, job scripts and summary scripts used for R27-R31")
    for f in ["algorithms/qrc.py", "algorithms/qrc-et-algo3.py", "multichain_env.py", "apply_eval_patch.py",
              "summarize_r28.py", "summarize_sweep.py", "summarize_r30.py", "summarize_r31.py",
              "wb_publish_runs.py", "wb_build_report.py", "tests/test_et_algo3.py"] + sorted(glob.glob("slurm/R2[7-9]*.sh")) \
            + sorted(glob.glob("slurm/R3*.sh")) + sorted(glob.glob("experiments/tabular_checks/*.py")):
        if os.path.exists(f):
            code.add_file(f, name=f)
    run.log_artifact(code)
    results = wandb.Artifact("qrc-et-analysis-r27-r31", type="results",
                             description="report_data.json and figures computed from all R27-R31 run CSVs")
    results.add_file(os.path.join(args.out, "report_data.json"), name="report_data.json")
    for key, path in figs.items():
        results.add_file(path, name="figures/" + os.path.basename(path))
    run.log_artifact(results)
    run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entity", default="mubbashirahmed")
    ap.add_argument("--project", default="qrc-et-algo3-minatar")
    ap.add_argument("--prefixes", nargs="+", default=["R27", "R28", "R29", "R30", "R31"])
    ap.add_argument("--out", default="wandb_upload")
    ap.add_argument("--stride", type=int, default=5000, help="history thinning for runs with >2000 rows")
    ap.add_argument("--analysis-only", action="store_true")
    ap.add_argument("--no-analysis", action="store_true", help="only publish the runs (skip the R27-R31 analysis run)")
    ap.add_argument("--list", action="store_true", help="only list the runs that would be published")
    args = ap.parse_args()

    paths = discover(args.prefixes)
    by_exp = defaultdict(int)
    for p in paths:
        by_exp[os.path.basename(os.path.dirname(p)).split("__")[-2].split("_")[0]] += 1
    print("runs found:", dict(by_exp))
    if args.list:
        for p in paths:
            print(" ", p)
        return
    if not args.analysis_only:
        publish(paths, args)
    if args.no_analysis:
        print(f"\nDone. Upload with:  cd {args.out} && wandb sync wandb/offline-run-*")
        return
    data, figs = analyse(args.out)
    print("coverage:", json.dumps(data["coverage"]))
    log_analysis(args, data, figs)
    print(f"\nDone. Tables: {args.out}/report_data.json   Figures: {args.out}/figures/")
    print(f"Upload with:  cd {args.out} && wandb sync --sync-all")


if __name__ == "__main__":
    main()
