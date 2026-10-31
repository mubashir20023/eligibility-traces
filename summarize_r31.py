"""Summarize R31 (Theta target x Theta learning rate) against R30's best QRC. Lower is better.

Per run: AUC = mean value_rmse over the run; TRACE = mean et_trace_loss over the last 10% of steps
(how well Theta fits its target at the end). Usage (from stream-rep-rl):  python summarize_r31.py
"""
import csv, glob, re
from collections import defaultdict
import numpy as np

TOTAL = 100_000
ENVS = ["m1-n4-s0.0", "m32-n4-s0.0", "m1-n4-s1.0", "m32-n4-s1.0"]


def load(pattern, regex):
    out = defaultdict(dict)
    for p in glob.glob(pattern):
        m = re.search(regex, p)
        if not m:
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if not rows or float(rows[-1]["global_step"]) < 0.99 * TOTAL:
            continue
        steps = np.array([float(r["global_step"]) for r in rows])
        err = np.array([float(r["value_rmse"]) for r in rows])
        err = np.where(np.isfinite(err), err, 10.0)
        tl = np.array([float(r.get("et_trace_loss", "nan")) for r in rows])
        out[m.groups()[:-1]][int(m.groups()[-1])] = (err.mean(), np.nanmean(tl[steps > 0.9 * TOTAL]) if np.isfinite(tl).any() else np.nan)
    return out


r31 = load("logs/multichain_env:MultiChainPred-*__R31_*__*/training_data.csv",
           r"MultiChainPred-(m\d+-n\d+-s[\d.]+)-v0__R31_et_(e_w|raw)_etlr([\d.]+)_lr([\de.-]+)__(\d+)/")
r30 = load("logs/multichain_env:MultiChainPred-*__R30_qrc_*__*/training_data.csv",
           r"MultiChainPred-(m\d+-n\d+-s[\d.]+)-v0__R30_(qrc_lam[\d]+)_lr([\de.-]+)__(\d+)/")
r30et = load("logs/multichain_env:MultiChainPred-*__R30_et_eta0_*__*/training_data.csv",
             r"MultiChainPred-(m\d+-n\d+-s[\d.]+)-v0__R30_(et_eta0)_lr([\de.-]+)__(\d+)/")

for env in ENVS:
    qrc = [(np.mean([v[0] for v in d.values()]), k, d) for k, d in r30.items() if k[0] == env and len(d) == 5]
    if not qrc:
        print(f"\n=== {env}: R30 QRC runs missing"); continue
    q_auc, q_key, q_runs = min(qrc, key=lambda x: x[0])
    print(f"\n=== {env}   best R30 QRC: {q_key[1]} lr {q_key[2]}  AUC {q_auc:.4f}")
    print(f"{'Theta target':<14}{'et_lr':>7}{'q_lr':>7}{'AUC':>10}{'vs QRC':>9}{'ET better':>11}{'trace loss':>12}")
    rows = []
    for (e, tgt, etlr, qlr), d in sorted(r31.items()):
        if e != env:
            continue
        auc = np.mean([v[0] for v in d.values()])
        common = sorted(set(d) & set(q_runs))
        wins = sum(d[s][0] < q_runs[s][0] for s in common)
        rows.append((auc, tgt, etlr, qlr, wins, len(common), np.nanmean([v[1] for v in d.values()])))
    for auc, tgt, etlr, qlr, wins, n, tl in sorted(rows):
        print(f"{tgt:<14}{etlr:>7}{qlr:>7}{auc:>10.4f}{100 * (auc / q_auc - 1):>+8.1f}%{f'{wins}/{n}':>11}{tl:>12.4f}")
    for (e, _, qlr), d in sorted(r30et.items()):
        if e == env and len(d) == 5:
            print(f"{'(R30 e_w, et_lr 0.1)':<28}{qlr:>7}{np.mean([v[0] for v in d.values()]):>10.4f}"
                  f"{'':>20}{np.nanmean([v[1] for v in d.values()]):>12.4f}")
print("\nWhat to look for: a target/et_lr whose trace loss is much lower AND whose AUC beats QRC (negative %),"
      " especially at m32 with sigma 1.0.")
