"""Audit and export the completed two-method screen without private machine paths."""
import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]
METHODS = {"qrc": "QRC", "et_state_nlms": "Experimental QRC-ET"}
FIELDS = ["global_step", "avg_return", "td_squared_error", "et_trace_loss", "elapsed_seconds"]
CONFIG_FIELDS = ["seed", "q_lr", "gamma", "lamda", "total_timesteps", "explore_frac",
                 "et_eta", "et_lr", "et_algo3", "et_algo3_target", "et_algo3_lag",
                 "et_paired_rng", "et_zero_init", "et_normalize_grad",
                 "et_action_conditioned", "et_post_update", "et_warmup_steps"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def curve_average(rows, budget):
    area, previous_step, previous_return = 0.0, 0, 0.0
    for row in rows:
        step, value = int(row["global_step"]), float(row["avg_return"])
        area += (step - previous_step) * (value + previous_return) / 2
        previous_step, previous_return = step, value
    return area / budget


def metrics(rows, budget):
    tail = [r for r in rows if .9 * budget < int(r["global_step"]) <= budget]
    return {"score": statistics.mean(float(r["avg_return"]) for r in tail),
            "curve_average": curve_average(rows, budget),
            "td_squared": statistics.mean(float(r["td_squared_error"]) for r in tail),
            "trace_loss": (statistics.mean(float(r["et_trace_loss"]) for r in tail)
                           if tail[0]["et_trace_loss"] else None)}


def export_run(folder, game, method, budget, out):
    source = folder / method
    result, config = read_json(source / "result.json"), read_json(source / "config.json")
    rows, episodes = read_csv(source / "training_data.csv"), read_csv(source / "episodes.csv")
    assert result["status"] == "complete" and result["steps"] == budget
    assert [int(r["global_step"]) for r in rows] == list(range(1000, budget + 1, 1000))
    assert config["seed"] == 601 and config["q_lr"] == .002
    assert math.isclose(config["total_timesteps"] * config["explore_frac"], 500000)
    assert config["gamma"] == .99 and config["lamda"] == .8
    if method != "qrc":
        assert config["et_zero_init"] and config["et_normalize_grad"]
        assert config["et_paired_rng"] and config["et_algo3"] and config["et_algo3_lag"]
        assert config["et_algo3_target"] == "e_w" and config["et_eta"] == .3
        assert not config["et_action_conditioned"] and not config["et_post_update"]
        assert config["et_warmup_steps"] == 0
    ends = [int(r["end_step"]) for r in episodes]
    rewards = [float(r["true_return"]) for r in episodes]
    for row in rows:
        count = bisect.bisect_right(ends, int(row["global_step"]))
        assert count == int(row["episodes"])
        expected = statistics.mean(rewards[max(0, count - 100):count]) if count else 0.0
        assert math.isclose(float(row["avg_return"]), expected, rel_tol=1e-6, abs_tol=1e-6)
        for key in FIELDS[1:]:
            if row[key]:
                assert math.isfinite(float(row[key]))
    calculated = metrics(rows, budget)
    assert math.isclose(calculated["score"], result["final_window_score"], abs_tol=1e-10)
    assert math.isclose(calculated["td_squared"], result["sampled_tail_td_squared"], abs_tol=1e-10)
    if method != "qrc":
        assert math.isclose(calculated["trace_loss"], result["sampled_tail_trace_loss"], abs_tol=1e-10)
    rid = f"localet261015_{game.lower()}_{budget // 1000000}m_{method}"
    path = out / "data" / f"{rid}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({key: row[key] for key in FIELDS} for row in rows)
    manifest = read_json(folder / "manifest.json")
    return {"id": rid, "game": game, "method": METHODS[method], "variant": method,
            "budget": budget, "seed": 601, **calculated,
            "minutes": result["total_seconds"] / 60, "episodes": len(episodes),
            "started_utc": result["started_utc"], "finished_utc": result["finished_utc"],
            "source_sha256": result["source_sha256"],
            "initialization": read_json(source / "initialization.json"),
            "config": {k: config[k] for k in CONFIG_FIELDS if k in config},
            "packages": manifest["packages"], "data_file": path.relative_to(out).as_posix(),
            "data_sha256": digest(path), "raw_training_sha256": digest(source / "training_data.csv"),
            "raw_episodes_sha256": digest(source / "episodes.csv"),
            "audit": {"rolling_returns_reconciled": len(rows), "episode_records_checked": len(episodes)}}


def validate_export(out):
    evidence = read_json(out / "evidence.json")
    runs = evidence["runs"]
    assert len(runs) == 12 and len({r["id"] for r in runs}) == 12
    assert {r["variant"] for r in runs} == set(METHODS)
    for run in runs:
        path = out / run["data_file"]
        assert digest(path) == run["data_sha256"]
        for key, value in metrics(read_csv(path), run["budget"]).items():
            assert value is None and run[key] is None or math.isclose(value, run[key], abs_tol=1e-10)
        partner = next(r for r in runs if r["game"] == run["game"]
                       and r["budget"] == run["budget"] and r["variant"] != run["variant"])
        assert partner["initialization"] == run["initialization"]
    for method in METHODS:
        short = next(r for r in runs if r["game"] == "Breakout" and r["budget"] == 1000000 and r["variant"] == method)
        long = next(r for r in runs if r["budget"] == 3000000 and r["variant"] == method)
        a, b = read_csv(out / short["data_file"]), read_csv(out / long["data_file"])[:1000]
        assert all(x[k] == y[k] for x, y in zip(a, b) for k in FIELDS if k != "elapsed_seconds")
        assert short["initialization"] == long["initialization"]
        assert short["source_sha256"] == long["source_sha256"] and short["packages"] == long["packages"]
    text = json.dumps(evidence)
    assert "et_original" not in text and "C:\\" not in text and "G:\\" not in text
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--freeway-root", type=Path, required=True)
    parser.add_argument("--breakout-root", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for game in GAMES:
        folder = args.freeway_root if game == "Freeway" else args.screen_root / game
        runs.extend(export_run(folder, game, method, 1000000, HERE) for method in METHODS)
    runs.extend(export_run(args.breakout_root, "Breakout", method, 3000000, HERE) for method in METHODS)
    candidate = ROOT / "algorithms/qrc-et-algo3-lab.py"
    assert digest(candidate) == next(r["source_sha256"] for r in runs if r["variant"] != "qrc")
    assert digest(ROOT / "algorithms/qrc.py") == runs[0]["source_sha256"]
    evidence = {"schema": 1, "as_of_utc": max(r["finished_utc"] for r in runs),
                "scope": "Two methods only; one matched seed 601; five 1M-step pairs and one 3M-step Breakout pair",
                "score_definition": "Mean rolling-100-episode training return over 0.9*budget < step <= budget",
                "curve_average_definition": "Trapezoidal integral of logged return divided by budget, with return zero at step zero",
                "diagnostic_definition": "Mean of 1000-step diagnostic samples in the final 10% window, not all transitions",
                "runtime_definition": "Measured run wall time including initialization/compilation; uncontrolled PC load",
                "reproducibility": "Breakout first-1M non-timing log fields match exactly; not independent seed replicates",
                "runs": runs}
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False), encoding="ascii")
    validate_export(HERE)
    print(json.dumps({"passed": True, "runs": len(runs),
                      "rolling_returns_checked": sum(r["audit"]["rolling_returns_reconciled"] for r in runs),
                      "episode_records_checked": sum(r["episodes"] for r in runs)}))


if __name__ == "__main__":
    main()
