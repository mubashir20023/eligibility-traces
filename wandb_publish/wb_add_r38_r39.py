"""Add the R38 (non-convolutional networks) + R39 / R39b (pre-trained representation) section to the existing report,
as the newest follow-up: inserted right before the R36-R37 section, same layout (compact tables + figure pairs).

Usage (repo root, ~/wbenv):
    ~/wbenv/bin/python wb_add_r38_r39.py --dry-run     # build tables + figures locally, change nothing
    ~/wbenv/bin/python wb_add_r38_r39.py               # log figures run, back up the report, insert, save
"""
import argparse, os, re
from collections import defaultdict

import numpy as np

from wb_publish_runs import paired, mean_sem
from wb_reorganize import fig_pair, md, block_text, REPORT_URL, GITHUB
from wb_add_r36_r37 import read, GAMES

TAG = "FOLLOWUP-R38-R39-FIGS"
FIGDIR = "report_figures"
NEXT_SECTION = "Equal-Budget Tuning on All Five MinAtar Games"     # R36-R37 heading we insert in front of
GAME_COLORS = dict(zip(GAMES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]))
SIG = lambda c: c is not None and (c["ci_low"] > 0 or c["ci_high"] < 0)


def scores(d):
    return {s: v["score"] for s, v in d.items()}


def fmt(d):
    mu, se = mean_sem(scores(d).values())
    return f"{mu:.1f} ± {se:.1f}"


def final_table(R):
    """R = read('R38mlpF'): best QRC vs best QRC-ET per game on 10 fresh seeds."""
    rows, out = [], {}
    for g in GAMES:
        q = next((d for (gg, c), d in R.items() if gg == g and c.startswith("qrc_best")), None)
        e = next((d for (gg, c), d in R.items() if gg == g and c.startswith("et_best")), None)
        if not q or not e:
            continue
        c = paired(scores(e), scores(q))
        out[g] = (q, e, c)
        rows.append([g, fmt(q), fmt(e), f"**{c['diff']:+.1f} [{c['ci_low']:+.1f}, {c['ci_high']:+.1f}]**" if SIG(c)
                     else f"{c['diff']:+.1f} [{c['ci_low']:+.1f}, {c['ci_high']:+.1f}]", f"{c['a_higher']}/{c['n']}"])
    return md(["Game", "Best QRC", "Best QRC-ET", "ET − QRC (95% CI)", "ET higher on"], rows), out


def tuning_table(R):
    """Best setting per method per game (3 tuning seeds) and divergences split by lambda."""
    rows, div = [], defaultdict(lambda: [0, 0])
    for g in GAMES:
        best = {}
        for (gg, c), d in R.items():
            m = re.match(r"(qrc|et)_lam([\d.]+)", c)
            if gg != g or not m:
                continue
            mu = np.mean(list(scores(d).values()))
            if m.group(1) not in best or mu > best[m.group(1)][0]:
                best[m.group(1)] = (mu, c)
            for v in d.values():
                k = (m.group(1), m.group(2))
                div[k][0] += v["diverged"]; div[k][1] += 1
        if "qrc" in best and "et" in best:
            rows.append([g, f"{best['qrc'][0]:.1f} ({best['qrc'][1].replace('qrc_', '')})",
                         f"{best['et'][0]:.1f} ({best['et'][1].replace('et_', '')})", f"{best['et'][0] - best['qrc'][0]:+.1f}"])
    lams = sorted({k[1] for k in div}, key=float)
    drows = [[("QRC" if m == "qrc" else "QRC-ET")] + [f"{div[(m, l)][0]}/{div[(m, l)][1]}" if div[(m, l)][1] else "–" for l in lams]
             for m in ("qrc", "et")]
    return (md(["Game", "Best QRC (setting)", "Best QRC-ET (setting)", "ET − QRC"], rows),
            md(["Diverged runs"] + [f"λ={l}" for l in lams], drows))


def build(R):
    T, F, S = {}, {}, {}
    os.makedirs(FIGDIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- R39b: frozen features, identical settings
    pts, counts, rows39b = [], defaultdict(lambda: [0, 0, 0]), []
    for g in GAMES:
        for src in ("qrc", "et"):
            cells = defaultdict(dict)
            for (gg, c), d in R["R39b"].items():
                m = re.match(rf"(qrc|et)_src{src}_lam([\d.]+)_lr([\de.-]+)$", c)
                if gg == g and m:
                    cells[(m.group(2), m.group(3))][m.group(1)] = scores(d)
            if not cells:
                continue
            ref = max(np.mean(list(v["qrc"].values())) for v in cells.values() if "qrc" in v)
            better = worse = 0
            bq = be = -1e9
            for k, v in cells.items():
                if "qrc" not in v or "et" not in v:
                    continue
                qm, em = np.mean(list(v["qrc"].values())), np.mean(list(v["et"].values()))
                bq, be = max(bq, qm), max(be, em)
                c = paired(v["et"], v["qrc"])
                if SIG(c):
                    better += c["diff"] > 0; worse += c["diff"] < 0
                pts.append((g, qm / ref, em / ref))
            counts[g][0] += better; counts[g][1] += worse; counts[g][2] += len(cells)
            rows39b.append([g, "QRC's" if src == "qrc" else "QRC-ET's", f"{bq:.1f}", f"{be:.1f}", f"{better} / {worse} of {len(cells)}"])
    T["r39b"] = md(["Game", "Features from", "Best QRC", "Best QRC-ET", "ET significantly better / worse"], rows39b)
    tot = [sum(v[0] for v in counts.values()), sum(v[1] for v in counts.values()), sum(v[2] for v in counts.values())]
    S["r39b"] = tot
    if pts:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))
        ax = axes[0]
        for g in GAMES:
            p = np.array([(x, y) for gg, x, y in pts if gg == g])
            if len(p):
                ax.scatter(p[:, 0], p[:, 1], s=22, color=GAME_COLORS[g], label=g, alpha=.8)
        ax.plot([0, 1.15], [0, 1.15], "k--", lw=1)
        ax.set_xlim(-0.03, 1.15); ax.set_ylim(-0.03, 1.15)
        ax.set_xlabel("QRC score ÷ best QRC score"); ax.set_ylabel("QRC-ET score ÷ best QRC score")
        ax.set_title("R39b: every matched setting (frozen features)", fontsize=11)
        ax.grid(alpha=.3); ax.legend(fontsize=8, frameon=False)
        ax = axes[1]
        x = np.arange(len(GAMES))
        ax.bar(x - 0.2, [counts[g][0] for g in GAMES], 0.4, color="#2ca02c", label="QRC-ET significantly better")
        ax.bar(x + 0.2, [counts[g][1] for g in GAMES], 0.4, color="#d62728", label="QRC-ET significantly worse")
        ax.set_xticks(x); ax.set_xticklabels(GAMES, fontsize=9)
        ax.set_ylabel("settings (of 16 per game)")
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(f"R39b: significant differences ({tot[0]} better, {tot[1]} worse of {tot[2]})", fontsize=11)
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); F["r39b"] = os.path.join(FIGDIR, "r39b_matched.png"); fig.savefig(F["r39b"], dpi=150); plt.close(fig)

    # ---------------- R39: own tuned settings (confounded)
    rows39 = []
    for g in GAMES:
        cells = []
        for src in ("qrc", "et"):
            for mode in ("frz", "unf"):
                q = R["R39"].get((g, next((c for gg, c in R["R39"] if gg == g and c.startswith(f"qrc_src{src}_{mode}_")), "")))
                e = R["R39"].get((g, next((c for gg, c in R["R39"] if gg == g and c.startswith(f"et_src{src}_{mode}_")), "")))
                if q and e:
                    c = paired(scores(e), scores(q))
                    cells.append(f"**{c['diff']:+.1f}**" if SIG(c) else f"{c['diff']:+.1f}")
                else:
                    cells.append("–")
        rows39.append([g] + cells)
    T["r39"] = md(["Game", "QRC's features, frozen", "QRC's features, unfrozen", "QRC-ET's features, frozen", "QRC-ET's features, unfrozen"], rows39)

    # ---------------- R38: MLP and linear
    T["mlp_final"], fin_mlp = final_table(R["R38mlpF"])
    T["lin_final"], fin_lin = final_table(R["R38lin2F"])
    T["mlp_tune"], T["mlp_div"] = tuning_table(R["R38mlp"])
    T["lin_tune"], T["lin_div"] = tuning_table(R["R38lin2"])
    if fin_mlp or fin_lin:
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.0))
        for ax, (name, fin) in zip(axes, [("MLP (no conv)", fin_mlp), ("Linear, raw pixels", fin_lin)]):
            gs = [g for g in GAMES if g in fin]
            x = np.arange(len(gs))
            for off, idx, col, lab in [(-0.2, 0, "#1f77b4", "best QRC"), (0.2, 1, "#d62728", "best QRC-ET")]:
                ref = [np.mean(list(scores(fin[g][0]).values())) for g in gs]
                vals = [np.array(list(scores(fin[g][idx]).values())) / r for g, r in zip(gs, ref)]
                ax.bar(x + off, [v.mean() for v in vals], 0.4, yerr=[v.std(ddof=1) / np.sqrt(len(v)) for v in vals],
                       color=col, label=lab, capsize=3)
            ax.set_xticks(x); ax.set_xticklabels(gs, fontsize=9); ax.axhline(1, color="k", lw=.8, ls="--")
            ax.set_ylabel("score ÷ best QRC score"); ax.set_title(f"R38 {name}: 10 fresh seeds", fontsize=11)
            ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, frameon=False)
        fig.tight_layout(); F["r38_final"] = os.path.join(FIGDIR, "r38_final.png"); fig.savefig(F["r38_final"], dpi=150); plt.close(fig)
    if "Freeway" in fin_mlp:
        q, e, _ = fin_mlp["Freeway"]
        gx = np.arange(25_000, 5_000_001, 25_000)
        panels = [(f"R38 MLP Freeway: {lab}, each seed", gx,
                   {f"seed {s}": np.interp(gx, v["step"], v["ret"]) for s, v in sorted(d.items())},
                   "Environment steps", "rolling-100-episode return", {}) for lab, d in [("best QRC", q), ("best QRC-ET", e)]]
        F["r38_mlp_freeway"] = fig_pair(os.path.join(FIGDIR, "r38_mlp_freeway.png"), panels)
        S["mlp_fw_collapsed"] = sum(v["score"] < 1 for v in e.values())
    return T, F, S


def blocks(wr, T, F, S, entity, project):
    rs = wr.Runset(entity=entity, project=project, name="Figures | R38-R39", filters=f"Config('experiment') == '{TAG}'")
    M = lambda t: wr.MarkdownBlock(text=t)

    def fig(key, h=11):
        return wr.PanelGrid(runsets=[rs], panels=[wr.MediaBrowser(title="", media_keys=[f"fig/{key}"], num_columns=1,
                                                                  layout=wr.Layout(x=0, y=0, w=24, h=h))]) if key in F else None
    b, w, n = S.get("r39b", [0, 0, 0])
    overview = md(["Setting", "QRC-ET vs QRC"], [
        ["One-hot multi-chain (R33-R35)", "**Better**: 13-32% lower value error, every seed"],
        ["MinAtar conv, equal tuning (R36-R37)", "Level on all 5 games"],
        ["MinAtar MLP, no conv (R38)", "Worse on Freeway and SpaceInvaders, level elsewhere"],
        ["MinAtar linear, raw pixels (R38)", "Level (slightly worse on Asterix)"],
        ["Pre-trained conv, frozen, identical settings (R39b)", f"Level: significantly better in {b}, worse in {w} of {n} settings"]])
    out = [
        wr.HorizontalRule(),
        M("# Latest | Non-Convolutional Networks and Pre-Trained Representations (R38, R39)"),
        M("**Summary.** Peter's two suggestions to find out why QRC-ET does not improve on QRC on MinAtar: (1) networks "
          "without convolution, as on the multi-chain task; (2) copy the representation layers of the best agents, re-initialise "
          "the Q-head, clear the traces, freeze the representation, and compare again, to isolate whether QRC-ET helps the "
          "policy learning itself. **In every MinAtar variant QRC-ET is level with or slightly behind QRC.** The only clear gain "
          "remains the one-hot multi-chain task, where every state has its own feature; even a linear network on pixels shares "
          "weights across many states.\n\n" + overview),
        M("## Pre-Trained Representation, Frozen, Identical Settings (R39b)\n\n"
          "Conv layers copied from the best R37 QRC run or the best R37 QRC-ET run (so neither method only gets its own "
          "features), fresh Q-head, cleared traces, Θ and optimiser state; conv layers frozen. Both methods use exactly the same "
          "settings (learning rate 2.5e-4 / 5e-4 / 1e-3 / 2e-3 × λ 0.24 / 0.8; QRC-ET split form, η=0.3, Θ/Q ratio 100), "
          "5 seeds (501-505), 5M steps, 800 runs; differences paired over seeds.\n\n" + T["r39b"]),
        fig("r39b"),
        M("QRC-ET's few significant wins come from settings where QRC collapsed or neither method learned; at the settings that "
          "work well (λ=0.24, low learning rate) the differences are small and slightly favour QRC. λ=0.8 on frozen features "
          "often fails for both methods (Freeway, Seaquest)."),
        M("## Pre-Trained Representation With Each Method's Own Settings (R39)\n\n"
          "First version of the same test, with each method at its own R37 learning rate and λ, frozen and unfrozen, 10 seeds "
          "(501-510), 400 runs. Cells: ET − QRC final score (bold = significant).\n\n" + T["r39"] +
          "\nThese differences are **confounded by the learning rate**: e.g. on Asterix QRC used lr 1e-3 and QRC-ET 5e-4, and "
          "QRC's score falls when it keeps training the features at the higher rate. With identical settings (R39b) the Asterix "
          "gap disappears (best 44.1 vs 43.5)."),
        M("## Networks Without Convolution: MLP (R38)\n\n"
          "Flatten → Dense(128) → Q-head, same LayerNorm / activation / initialisation as the conv network. Equal-budget tuning as "
          "R36 (24 settings each, 3 seeds), then each method's best setting on 10 fresh seeds (401-410).\n\n" + T["mlp_final"]),
        fig("r38_final"),
        fig("r38_mlp_freeway"),
        M(f"On Freeway, {S.get('mlp_fw_collapsed', '?')} of 10 QRC-ET seeds collapse to a return of about 0 (right); the seeds "
          "that learn match QRC. The best QRC-ET setting (λ=0.8) was chosen on 3 tuning seeds that all learned."),
        M("### MLP tuning (3 seeds)\n\n" + T["mlp_tune"] + "\n" + T["mlp_div"] +
          "\nAt the λ both methods had (0.8) the divergence rates are similar; QRC's extra divergences come from λ ≥ 0.9."),
        M("## Networks Without Convolution: Linear on Raw Pixels (R38)\n\n"
          "Q = W·x on the flattened 0/1 image, no bias. A first attempt with observation normalisation diverged almost everywhere "
          "(rare pixels become huge inputs and a linear network has no normalising layer), so it was re-run on raw pixels with "
          "learning rates 2.5e-3 to 2e-2 (R38lin2). Same equal-budget tuning, then 10 fresh seeds.\n\n" + T["lin_final"] +
          "\nScores are far below the conv network (Freeway ~20 vs ~58), and neither method learns Seaquest."),
        M("### Linear tuning (3 seeds)\n\n" + T["lin_tune"] + "\n" + T["lin_div"]),
        M("## Reproduction | R38-R39\n\n" + md(["Experiment", "Job script", "Runs"], [
            ["R38 MLP tuning / final", "slurm/R38_nonconv_tune.sh, R38_nonconv_final.sh (ARCH=mlp)", "720 / 100"],
            ["R38 linear, raw pixels: tuning / final", "same scripts (ARCH=linear, --time=60:00:00)", "720 / 100"],
            ["R39 pre-trained, own settings", "slurm/R39_transfer.sh (+ prepare_r39.py)", "400"],
            ["R39b pre-trained, frozen, identical settings", "slurm/R39b_frozen_matched.sh", "800"]]) +
          "\nNew agent options (apply_r38_r39_patch.py): `--net_arch mlp`, `--init_from` (copy all layers except the Q-head), "
          f"`--freeze_body`. Code: branch [cluster-r29]({GITHUB}), `RESULTS_R38_R39.md`."),
    ]
    return [x for x in out if x is not None]


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

    R = {p: read(p) for p in ["R38mlp", "R38mlpF", "R38lin2", "R38lin2F", "R39", "R39b"]}
    print("complete runs: " + ", ".join(f"{p} {sum(len(d) for d in r.values())}" for p, r in R.items()))
    T, F, S = build(R)
    for k, v in T.items():
        print(f"\n### {k}\n{v}")
    print("figures:", ", ".join(F.values()))
    new = blocks(wr, T, F, S, args.entity, args.project)
    if args.dry_run:
        print(f"\n{len(new)} blocks built (nothing changed).")
        return

    rep = wr.Report.from_url(args.url)
    bl = list(rep.blocks)
    texts = [block_text(b) for b in bl]
    if any("Non-Convolutional Networks and Pre-Trained Representations" in t for t in texts):
        raise SystemExit("The R38-R39 section is already in the report; nothing changed.")
    i = next((k for k, t in enumerate(texts) if NEXT_SECTION in t), None)
    if i is None:
        raise SystemExit("Could not find the R36-R37 section heading; nothing changed.")
    bl[i].text = bl[i].text.replace("# Latest | ", "# Follow-up | ")
    if i > 0 and type(bl[i - 1]).__name__ == "HorizontalRule":
        i -= 1
    if not args.no_log:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, name="FOLLOW-UP | R38-R39 report figures",
                         job_type="report-data", group="ANALYSIS", config={"experiment": TAG, "code": GITHUB})
        run.log({f"fig/{k}": wandb.Image(p) for k, p in F.items()})
        run.finish()
    backup = wr.Report.from_url(args.url)
    backup.title = backup.title + " (backup before R38-R39)"
    backup.save(clone=True)
    print("Backup of the current report:", backup.url)
    rep.blocks = bl[:i] + new + bl[i:]
    rep.width = "fluid"
    rep.save()
    print(f"Inserted {len(new)} blocks at position {i}. Updated report:", rep.url)


if __name__ == "__main__":
    main()
