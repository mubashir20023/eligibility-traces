# Expected traces in streaming QRC: R33-R35

## What changed in the code

| Flag (algorithms/qrc-et-algo3.py) | Meaning |
|---|---|
| `--no-et_algo3_lag` | Use this step's expected trace z(S_t) in the head update (Algorithm 3 uses it one step late). |
| `--et_lr` | Theta step size. Effective per-visit step is about `et_lr * 2 / F` for one-hot input (F = number of input features), so `et_lr = 0.1 * F / 2` gives beta ~ 0.1. |
| `--et_split` | Theta predicts only the past part of the trace; head update = exact current gradient + z_past(S_t). Stable form of the no-lag fix. |
| `--net_arch linear --no-obs_norm` (qrc.py and qrc-et-algo3.py) | One linear layer, no bias, zero init, raw one-hot input (tabular). Use `--et_head_name Dense_0`. |
| `--eval_values --snapshot_episodes ...` | Value RMSE against exact values, plus per-state snapshots and diagnostics (multi-chain only). |

The patches (`apply_eval_patch.py`, `apply_snapshot_patch.py`, `apply_linear_patch.py`, `apply_split_patch.py`) are already applied to the files in `algorithms/`; they are kept for the record and skip files that are already patched.

Environment: `multichain_env.py`, `MultiChainSparse-m{8,32}-n4-p{0.2,0.5,0.9}-v0` (start, one of m chains of length 4, shared bottleneck, reward +1 with probability p).

## One-hot multi-chain (value-error AUC vs best tuned QRC, lower is better)

| Task | R33 no-lag (10 seeds) | R34a no-lag (seeds 11-30) | R35a split (seeds 11-30) | Current code, with lag (R34a) |
|---|---|---|---|---|
| m8, p=0.2 | -13.2% | -12.7% (20/20) | **-13.0% (20/20)** | +74% |
| m8, p=0.9 | -16.2% | -14.5% (20/20) | **-16.1% (20/20)** | +94% |
| m32, p=0.2 | -14.7% | -13.8% (20/20) | **-16.3% (20/20)** | +79% |
| m32, p=0.9 | -23.9% | -24.2% (20/20) | **-32.0% (20/20)** | +63% |

(n/20 = seeds on which ET beats the best QRC; all p < 1e-13, paired t-test.)

## Conv network on the same task (R34b, 10 seeds)

Best QRC uses lambda = 0 on all four tasks. Every QRC-ET variant is 16-34% worse than it, but 10-17% better than QRC lambda = 0.9 at the same learning rate. Conv features collapse (trace diversity ~0, effective rank 61 -> 3-8), so the network already shares credit across chains and the expected trace has nothing to average.

## MinAtar (R34c, R35; 5 seeds 201-205, 5M steps, final score, higher is better)

| Game | Best QRC | Split ET, eta 0.3 | Split ET, eta 0 | No-lag ET (R34c) | Current code, eta 0.3 |
|---|---|---|---|---|---|
| Freeway | 55.3 (lambda 0.24) | **55.3** | 33.1 (4/10 runs collapse) | 6.4 (diverged 16/20) | 53.8 |
| Breakout | 14.4 (lambda 0.8) | **15.2** | 13.1 | 13.7 | 12.1 |
| Asterix | 30.7 (lambda 0.24) | 24.2 | 22.5 | 0.5 (diverged 18/20) | 22.8 |

Split ET never diverged (0/60 runs). With eta = 0, 4 of 10 Freeway runs learned and then collapsed to Q ~ 0 and return 0. ET was only run with lambda = 0.8; at matched lambda = 0.8 ET vs QRC is Freeway 55.3 vs 50.9, Breakout 15.2 vs 14.4, Asterix 24.2 vs 25.6. Five seeds: a screen.

## Summary

1. With features that keep states apart, compiled expected traces beat tuned QRC by 13-32% on every task and seed.
2. That needs the lag removed, a correctly sized Theta step, and (for stability with deep networks) the split form.
3. With conv features on MinAtar, split ET is stable and level with tuned QRC on Freeway and Breakout, and below the best (shorter-trace) QRC on Asterix. Conv features collapse states, so the representation limits what expected traces can add.

Summaries: `python summarize_mc.py R34a R35a`, `python summarize_mc.py R34b`, `python summarize_r34c.py R34c R35`.
