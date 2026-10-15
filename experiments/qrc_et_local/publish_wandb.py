"""Publish a NEW, two-method W&B report from the reviewed CSV export only."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import sys

from prepare_publication import GAMES, read_csv, validate_export

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TITLE = "QRC vs Experimental QRC-ET | Five-Game Screen and Breakout Follow-up"
ENTITY, PROJECT = "mubbashirahmed", "qrc-et-algo3-minatar"
COLORS = dict(zip(GAMES, ["#2563EB", "#B34883", "#00866A", "#C27813", "#1495AA"]))
BLUE, GREEN = "#2563EB", "#00866A"
DATASET = "local-qrc-et-20261015-v1"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, r)) + " |" for r in rows)])


def pair(runs, game, budget=1000000):
    selected = [r for r in runs if r["game"] == game and r["budget"] == budget]
    return tuple(next(r for r in selected if r["variant"] == method)
                 for method in ["qrc", "et_state_nlms"])


def pkt(iso):
    return datetime.fromisoformat(iso).astimezone(timezone(timedelta(hours=5))).strftime("%d %b %Y %H:%M")


def content(evidence, github):
    runs, parts = evidence["runs"], []
    def md(text):
        parts.append({"kind": "markdown", "text": text})
    def grid(rows, title, budget, overview=False):
        parts.append({"kind": "pair", "ids": [r["id"] for r in rows],
                      "title": title, "budget": budget, "overview": overview})
    md("## Executive Summary\n\n"
       "Experimental QRC-ET has higher final-window training return than QRC in **four of five games at 1M steps**, "
       "but this is a **single-seed screen**, not evidence of a reliable general improvement. "
       "Breakout remains worse at 3M steps: **10.81 versus 14.18**, a gap of **-3.38 points**.\n\n"
       "The tested change is **zero-initialized Theta plus normalized Theta gradients**, together. "
       "It is an Algorithm-3-style adaptation to QRC, not yet a faithful reproduction: current-action conditioning "
       "and post-Theta-update prediction were disabled. The earlier `e^w` target interpretation was already present.\n\n"
       "Only **QRC and this experimental QRC-ET** appear here. The historical research report, original-ET results, "
       "R27 and historical W&B runs are unchanged.")
    md("## All Five Games | 1M Steps\n\n"
       "Seed **601**, learning rate **0.002**, matched initialization and exploration keys. "
       "Final-window scores average the logged rolling-100-episode return over **900k < step <= 1M**. "
       "These are training returns, not independent policy evaluations.\n\n" +
       table(["Game", "QRC", "Experimental ET", "ET - QRC"], [
           [game, f"{q['score']:.2f}", f"{e['score']:.2f}", f"{e['score']-q['score']:+.2f}"]
           for game in GAMES for q, e in [pair(runs, game)]]) +
       "\n\nThe two overview graphs use the same game colors and axes. Raw game rewards have different meanings; "
       "they are shown together for orientation, not averaged or ranked across games.")
    grid([r for r in runs if r["budget"] == 1000000], "All five games", 1000000, True)
    md("## Matched Game Comparisons\n\nQRC is on the left, experimental QRC-ET on the right. "
       "Each pair uses identical step and reward scales. No confidence bands are shown because there is one seed.")
    for game in GAMES:
        q, e = pair(runs, game)
        note = {"Asterix": "Higher endpoint and curve average in this seed; generalization is untested.",
                "Breakout": "Both endpoint and curve average remain lower than QRC; the 3M follow-up below checks persistence.",
                "Freeway": "The endpoint gain is larger than the very small whole-budget curve-average gain.",
                "Seaquest": "Experimental ET is higher, but absolute returns remain low for both methods.",
                "SpaceInvaders": "The endpoint gain is accompanied by only a small whole-budget curve-average gain."}[game]
        md(f"### {game}\n\nFinal-window return: **{q['score']:.2f} vs {e['score']:.2f}** "
           f"(ET - QRC **{e['score']-q['score']:+.2f}**). Whole-budget curve averages: "
           f"**{q['curve_average']:.3f} vs {e['curve_average']:.3f}**. {note}")
        grid([q, e], game, 1000000)
    q, e = pair(runs, "Breakout", 3000000)
    milestones = []
    for budget in [1000000, 2000000, 3000000]:
        values = []
        for run in [q, e]:
            tail = [float(row["avg_return"]) for row in read_csv(HERE / run["data_file"])
                    if .9 * budget < int(row["global_step"]) <= budget]
            values.append(sum(tail) / len(tail))
        milestones.append([f"{budget // 1000000}M", f"{values[0]:.2f}", f"{values[1]:.2f}", f"{values[1]-values[0]:+.2f}"])
    md("## Breakout | The Gap Persists at 3M Steps\n\n"
       "Three-million-step runs started fresh with the same seed and unchanged 500k-step exploration schedule. "
       "The first 1M non-timing log fields match the earlier runs exactly; this is a reproducible extension, "
       "**not another independent seed**. Each row below compares equal budgets using its own last-10% window.\n\n" +
       table(["Budget", "QRC", "Experimental ET", "ET - QRC"], milestones) +
       "\n\nThe deficit grows from -1.16 at 1M to -3.38 at 3M. Longer training alone did not close it. "
       "This does not establish what will happen at 5M or with other seeds.")
    grid([q, e], "Breakout | 3M", 3000000)
    md("### Breakout Performance and Runtime\n\n" +
       table(["Method", "Final-window return", "Curve average", "Run minutes"], [
           [r["method"], f"{r['score']:.2f}", f"{r['curve_average']:.3f}", f"{r['minutes']:.1f}"] for r in [q, e]]) +
       "\n\nFinal-window sampled TD squared error is **0.052241 for QRC** versus **0.005895 for experimental ET**. "
       "ET trace regression loss is **0.173820**. Its lower sampled TD error does **not** imply better control; "
       "its training return is lower.")
    md("## Learning Efficiency, Diagnostics and Cost\n\n"
       "Curve average is the trapezoidal area under logged training return divided by the budget, with return zero at step zero. "
       "It complements the endpoint; it is not an independent evaluation metric.\n\n" +
       table(["Game", "QRC curve average", "ET curve average", "ET - QRC"], [
           [g, f"{a['curve_average']:.3f}", f"{b['curve_average']:.3f}", f"{b['curve_average']-a['curve_average']:+.3f}"]
           for g in GAMES for a, b in [pair(runs, g)]]) +
       "\n\nFinal-10%-window diagnostics sample one transition every 1k steps. They are not losses averaged over "
       "every transition, and different policies visit different state/reward distributions. QRC has no learned trace loss.\n\n" +
       table(["Game", "QRC TD squared", "ET TD squared", "ET trace loss"], [
           [g, f"{a['td_squared']:.6f}", f"{b['td_squared']:.6f}", f"{b['trace_loss']:.6f}"]
           for g in GAMES for a, b in [pair(runs, g)]]) +
       "\n\nMeasured wall time includes initialization and compilation. PC load varied; these are observed timings, "
       "not a controlled speed benchmark. W&B runtime is import time and must not be read as training time.\n\n" +
       table(["Game", "QRC minutes", "ET minutes"], [
           [g, f"{a['minutes']:.1f}", f"{b['minutes']:.1f}"] for g in GAMES for a, b in [pair(runs, g)]]))
    md("## What Changed and What Remains Different\n\n" +
       table(["Item", "Implemented behavior", "Interpretation"], [
           ["Shared representation", "Linear Theta uses Q-head features; ET loss does not train the trunk", "No separate deep trace network"],
           ["Algorithm 3 recursion", "Full final-layer e^w target; mixed trace enters subsequent QRC updates", "ET genuinely influences Q weights"],
           ["Line 15 target", "Use e^w, consistent with main-text p.5 and appendix explanation", "Already in supplied agent; apparent typo interpretation, not a new fix"],
           ["Paired RNG", "Same initial Q/h weights, environment seed and exploration key", "Prevents an extra Theta key split from changing the control stream"],
           ["New predictor initialization", "Theta starts at zero and then learns", "Tested jointly with normalization; effect not isolated"],
           ["New predictor update", "Normalize Theta gradients by feature energy", "Optimization variant, not demonstrated mechanism accuracy"],
           ["Current action input", "Disabled in tested candidate", "Predicting all action columns is not conditioning on current action"],
           ["Prediction ordering", "Pre-Theta-update prediction in tested candidate", "Paper places the mixture after Theta update"],
           ["Base learner", "Retains QRC gradient correction; substitutes head-kernel trace only", "Adaptation, not exact paper QET reproduction"],
       ]) +
       "\n\nReference: [Expected Eligibility Traces, pp.5 and 15](https://arxiv.org/pdf/2007.01839). "
       "The apparent line-15 typo is inferred from internal consistency, not an author-confirmed erratum.")
    md("## Progress and Next Decision\n\n" +
       table(["Stage", "Change or question", "Observed outcome / status"], [
           ["Earlier research", "Eta=1 control and shared-key splitting", "Research log records paired-RNG correction; old report retained separately"],
           ["Local implementation checks", "Zero initialization and feature-energy normalization", "Candidate selected for bounded screening; no proof of expected-trace mechanism"],
           ["14 Oct: five-game 1M screen", "One seed, same rate, matched starts", "Higher endpoints in four games; Breakout remains behind"],
           ["15 Oct: Breakout 3M follow-up", "Same source and exploration schedule", "Exact first-1M reproduction; endpoint gap remains -3.38"],
           ["Proposed next ablations", "Current-action conditioning, prediction ordering, then both", "Not trained in these reported runs; must be tested against unchanged controls"],
           ["Before a broad claim", "Held-out seeds, per-method rate tuning, full budgets and damping controls", "Still missing; no claim that historical weak seeds are fixed"],
       ]) +
       "\n\nThe next useful question is whether action conditioning or prediction timing improves returns, "
       "rather than merely reducing trace regression loss. Keep R27 and this frozen candidate separate. "
       "A short-lambda or zero-Theta control is needed to distinguish expected-trace learning from damping.")
    md("## Reproduction and Evidence\n\n"
       f"[Reviewed code, CSVs and instructions]({github}). "
       "The new agent filename is `algorithms/qrc-et-algo3-lab.py`; its flags are explicit in the reproduction instructions. "
       "Do not overwrite the cluster's R27 agent.\n\n"
       "All 12 imported result histories are complete. The export independently reconciled **16,000 rolling-return "
       "records with 216,716 episode records**. Counts include overlapping Breakout prefixes and are audit-work counts, "
       "not independent samples. Exported CSVs retain every 1k-step sample. W&B plots receive every 5k-step sample "
       "without additional smoothing; scores and diagnostics use the full 1k-step CSVs.\n\n"
       "All runs use CPU, seed 601, q_lr=0.002, gamma=0.99 and lambda=0.8. ET uses eta=0.3, Theta lr=0.1, "
       "paired RNG, e^w target and the Algorithm-3 lag. Zero initialization and normalization are enabled; "
       "action conditioning, warmup and post-update prediction are disabled.\n\n"
       f"Evidence cutoff: **{pkt(evidence['as_of_utc'])} Pakistan time (UTC+5)**. "
       "Per-run start/finish times, training minutes, initialization/source/data hashes and package versions "
       "are preserved in `evidence.json`. No raw returns are averaged across games. No significance test or confidence "
       "interval is justified by this single-seed, fixed-rate screen.")
    return parts


def build_blocks(wr, evidence, github):
    runs = {r["id"]: r for r in evidence["runs"]}
    blocks = []
    for part in content(evidence, github):
        if part["kind"] == "markdown":
            blocks.append(wr.MarkdownBlock(part["text"]))
            continue
        selected = [runs[rid] for rid in part["ids"]]
        rs = wr.Runset(entity=ENTITY, project=PROJECT, name=part["title"],
                        filters="ID in " + repr(part["ids"]),
                        order=[wr.OrderBy(wr.Config("Game"), ascending=True),
                               wr.OrderBy(wr.Config("Method"), ascending=True)],
                        visible_columns=["config:Game.value", "config:Method.value", "config:Seed.value"],
                        lock_columns=True)
        panels = []
        maximum = max(float(row["avg_return"]) for r in selected for row in read_csv(HERE / r["data_file"]))
        ymax = 30 if part["overview"] else max(1, math.ceil(maximum * 1.12 * 10) / 10)
        for index, (variant, label) in enumerate([("qrc", "QRC"), ("et_state_nlms", "Experimental QRC-ET")]):
            metric = "QRC return" if variant == "qrc" else "Experimental ET return"
            members = [r for r in selected if r["variant"] == variant]
            line_titles = {f"{r['id']}:{metric}": r["game"] if part["overview"] else f"{r['game']} | {label}" for r in members}
            colors = {f"{r['id']}:{metric}": COLORS[r["game"]] if part["overview"] else (BLUE if index == 0 else GREEN) for r in members}
            panels.append(wr.LinePlot(title=f"{part['title']} | {label}", x="Environment steps", y=[metric],
                           range_x=(0, part["budget"]), range_y=(0, ymax),
                           title_x="Environment steps", title_y="Rolling-100-episode training return",
                           aggregate=False, smoothing_type="none", ignore_outliers=False,
                           max_runs_to_show=12, font_size="large", legend_position="south",
                           line_titles=line_titles, line_colors=colors,
                           layout=wr.Layout(x=index * 12, y=0, w=12, h=11)))
        blocks.append(wr.PanelGrid(runsets=[rs], hide_run_sets=True, panels=panels))
    return blocks


def upload(wandb, run):
    settings = wandb.Settings(quiet=True, console="off", disable_code=True, disable_git=True,
                              x_disable_stats=True, x_disable_meta=True, x_disable_machine_info=True)
    config = {"dataset": DATASET, "Game": run["game"], "Method": run["method"],
              "Seed": run["seed"], "Budget": run["budget"], "Learning rate": .002,
              "Source SHA256": run["source_sha256"], "Data SHA256": run["data_sha256"],
              "Training started UTC": run["started_utc"], "Training finished UTC": run["finished_utc"],
              "imported_result": True, **run["config"]}
    with wandb.init(entity=ENTITY, project=PROJECT, id=run["id"], name=run["game"],
                    job_type="audited-local-result", group=f"Local {run['budget']//1000000}M | {run['game']}",
                    config=config, tags=[DATASET, run["method"], f"{run['budget']//1000000}M"],
                    notes="Audited completed training result import. W&B runtime is IMPORT time, not training time.",
                    settings=settings, reinit="create_new", resume="never") as remote:
        metric = "QRC return" if run["variant"] == "qrc" else "Experimental ET return"
        for row in read_csv(HERE / run["data_file"]):
            step = int(row["global_step"])
            if step % 5000 and step != run["budget"]:
                continue
            payload = {"Environment steps": step, metric: float(row["avg_return"]),
                       "Training return": float(row["avg_return"]),
                       "Training minutes": float(row["elapsed_seconds"]) / 60,
                       "Sampled TD squared": float(row["td_squared_error"])}
            if row["et_trace_loss"]:
                payload["Trace regression loss"] = float(row["et_trace_loss"])
            remote.log(payload)
        remote.summary.update({"Final-window return": run["score"], "Curve average": run["curve_average"],
                               "Measured training minutes": run["minutes"], "Budget steps": run["budget"],
                               "Sampled tail TD squared": run["td_squared"], "Audited complete": True,
                               **({"Sampled tail trace loss": run["trace_loss"]} if run["trace_loss"] is not None else {})})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github", required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--sdk-dir", type=Path)
    args = parser.parse_args()
    evidence = validate_export(HERE)
    sections = content(evidence, args.github)
    (HERE / "REPORT.md").write_text("# " + TITLE + "\n\n" + "\n\n".join(
        p["text"] if p["kind"] == "markdown" else f"[Interactive paired graphs: {p['title']}]"
        for p in sections), encoding="ascii")
    if not args.push:
        print("Prepared report narrative and verified 12 complete two-method results.")
        return
    if args.sdk_dir:
        sys.path.insert(0, str(args.sdk_dir.resolve()))
    if args.key_file:
        os.environ["WANDB_API_KEY"] = args.key_file.read_text().strip().strip("\"'")
    os.environ.update(WANDB_SILENT="true", WANDB_MODE="online", WANDB_DISABLE_CODE="true")
    import wandb
    import wandb_workspaces.reports.v2 as wr
    from wandb_workspaces._graphql import execute_graphql
    from wandb_workspaces.reports.v2 import gql
    from wandb_workspaces.reports.v2.interface import _get_api
    blocks = build_blocks(wr, evidence, args.github)
    for block in blocks:
        block._to_model()
    api = wandb.Api()
    state_path = HERE / "publication.json"
    prior = read_json(state_path) if state_path.exists() else None
    prior_ids = prior["run_ids"] if prior else []
    for run in evidence["runs"]:
        try:
            existing = api.run(f"{ENTITY}/{PROJECT}/{run['id']}")
        except wandb.errors.CommError as error:
            if "Could not find run" not in str(error) and "not found" not in str(error).lower():
                raise
            existing = None
        if existing is not None:
            assert existing.config.get("dataset") == DATASET
            assert existing.config.get("Data SHA256") == run["data_sha256"]
            assert existing.summary.get("Audited complete") is True
            print("Reusing audited import", run["id"], flush=True)
        else:
            upload(wandb, run)
            print("Imported", run["game"], run["method"], run["budget"], flush=True)
    report = wr.Report.from_url(prior["url"]) if prior else wr.Report(
        entity=ENTITY, project=PROJECT, title=TITLE,
        description="Matched-seed local screen: five games at 1M steps and Breakout at 3M; QRC and experimental ET only.", width="fluid")
    assert not prior or set(prior_ids) == {r["id"] for r in evidence["runs"]}
    report.title, report.width = TITLE, "fluid"
    report.blocks = blocks
    report.save()
    canonical = report.url.replace("\\", "/")
    saved = execute_graphql(_get_api(), gql.view_report, {"reportId": report.id})["view"]
    spec = json.loads(saved["spec"]) if isinstance(saved["spec"], str) else saved["spec"]
    assert spec["width"] == "fluid"
    remote = wr.Report.from_url(canonical)
    grids = [block for block in remote.blocks if isinstance(block, wr.PanelGrid)]
    assert len(grids) == 7 and all(g.hide_run_sets and len(g.panels) == 2 for g in grids)
    selected = " ".join(str(rs.filters) for grid in grids for rs in grid.runsets)
    assert all(run["id"] in selected for run in evidence["runs"]) and "et_original" not in selected
    verification_api = wandb.Api()
    for run in evidence["runs"]:
        imported = verification_api.run(f"{ENTITY}/{PROJECT}/{run['id']}")
        assert imported.config["Method"] == run["method"] and imported.config["Seed"] == 601
        assert imported.config["Data SHA256"] == run["data_sha256"]
        assert math.isclose(imported.summary["Final-window return"], run["score"], abs_tol=1e-9)
        assert imported.summary["Audited complete"] is True
    forge = canonical.replace("https://wandb.ai/", "https://forge.coreweave.com/wandb/")
    state = {"url": canonical, "forge_url": forge, "report_id": report.id, "github": args.github,
             "run_ids": [r["id"] for r in evidence["runs"]], "run_count": 12,
             "panel_count": 14, "width": spec["width"], "verified": True}
    state_path.write_text(json.dumps(state, indent=2), encoding="ascii")
    print(json.dumps(state, indent=2), flush=True)


def read_json(path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
