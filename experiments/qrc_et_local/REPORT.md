# QRC vs Experimental QRC-ET | Five-Game Screen and Breakout Follow-up

## Executive Summary

Experimental QRC-ET has higher final-window training return than QRC in **four of five games at 1M steps**, but this is a **single-seed screen**, not evidence of a reliable general improvement. Breakout remains worse at 3M steps: **10.81 versus 14.18**, a gap of **-3.38 points**.

The tested change is **zero-initialized Theta plus normalized Theta gradients**, together. It is an Algorithm-3-style adaptation to QRC, not yet a faithful reproduction: current-action conditioning and post-Theta-update prediction were disabled. The earlier `e^w` target interpretation was already present.

Only **QRC and this experimental QRC-ET** appear here. The historical research report, original-ET results, R27 and historical W&B runs are unchanged.

## All Five Games | 1M Steps

Seed **601**, learning rate **0.002**, matched initialization and exploration keys. Final-window scores average the logged rolling-100-episode return over **900k < step <= 1M**. These are training returns, not independent policy evaluations.

| Game | QRC | Experimental ET | ET - QRC |
| --- | --- | --- | --- |
| Asterix | 3.44 | 4.15 | +0.71 |
| Breakout | 9.67 | 8.51 | -1.16 |
| Freeway | 22.08 | 23.82 | +1.74 |
| Seaquest | 0.15 | 0.51 | +0.36 |
| SpaceInvaders | 23.71 | 25.66 | +1.96 |

The two overview graphs use the same game colors and axes. Raw game rewards have different meanings; they are shown together for orientation, not averaged or ranked across games.

[Interactive paired graphs: All five games]

## Matched Game Comparisons

QRC is on the left, experimental QRC-ET on the right. Each pair uses identical step and reward scales. No confidence bands are shown because there is one seed.

### Asterix

Final-window return: **3.44 vs 4.15** (ET - QRC **+0.71**). Whole-budget curve averages: **1.692 vs 2.048**. Higher endpoint and curve average in this seed; generalization is untested.

[Interactive paired graphs: Asterix]

### Breakout

Final-window return: **9.67 vs 8.51** (ET - QRC **-1.16**). Whole-budget curve averages: **5.135 vs 4.719**. Both endpoint and curve average remain lower than QRC; the 3M follow-up below checks persistence.

[Interactive paired graphs: Breakout]

### Freeway

Final-window return: **22.08 vs 23.82** (ET - QRC **+1.74**). Whole-budget curve averages: **9.429 vs 9.471**. The endpoint gain is larger than the very small whole-budget curve-average gain.

[Interactive paired graphs: Freeway]

### Seaquest

Final-window return: **0.15 vs 0.51** (ET - QRC **+0.36**). Whole-budget curve averages: **0.252 vs 0.444**. Experimental ET is higher, but absolute returns remain low for both methods.

[Interactive paired graphs: Seaquest]

### SpaceInvaders

Final-window return: **23.71 vs 25.66** (ET - QRC **+1.96**). Whole-budget curve averages: **14.832 vs 14.951**. The endpoint gain is accompanied by only a small whole-budget curve-average gain.

[Interactive paired graphs: SpaceInvaders]

## Breakout | The Gap Persists at 3M Steps

Three-million-step runs started fresh with the same seed and unchanged 500k-step exploration schedule. The first 1M non-timing log fields match the earlier runs exactly; this is a reproducible extension, **not another independent seed**. Each row below compares equal budgets using its own last-10% window.

| Budget | QRC | Experimental ET | ET - QRC |
| --- | --- | --- | --- |
| 1M | 9.67 | 8.51 | -1.16 |
| 2M | 12.52 | 10.51 | -2.01 |
| 3M | 14.18 | 10.81 | -3.38 |

The deficit grows from -1.16 at 1M to -3.38 at 3M. Longer training alone did not close it. This does not establish what will happen at 5M or with other seeds.

[Interactive paired graphs: Breakout | 3M]

### Breakout Performance and Runtime

| Method | Final-window return | Curve average | Run minutes |
| --- | --- | --- | --- |
| QRC | 14.18 | 10.153 | 75.5 |
| Experimental QRC-ET | 10.81 | 8.283 | 97.4 |

Final-window sampled TD squared error is **0.052241 for QRC** versus **0.005895 for experimental ET**. ET trace regression loss is **0.173820**. Its lower sampled TD error does **not** imply better control; its training return is lower.

## Learning Efficiency, Diagnostics and Cost

Curve average is the trapezoidal area under logged training return divided by the budget, with return zero at step zero. It complements the endpoint; it is not an independent evaluation metric.

| Game | QRC curve average | ET curve average | ET - QRC |
| --- | --- | --- | --- |
| Asterix | 1.692 | 2.048 | +0.356 |
| Breakout | 5.135 | 4.719 | -0.416 |
| Freeway | 9.429 | 9.471 | +0.042 |
| Seaquest | 0.252 | 0.444 | +0.193 |
| SpaceInvaders | 14.832 | 14.951 | +0.119 |

Final-10%-window diagnostics sample one transition every 1k steps. They are not losses averaged over every transition, and different policies visit different state/reward distributions. QRC has no learned trace loss.

| Game | QRC TD squared | ET TD squared | ET trace loss |
| --- | --- | --- | --- |
| Asterix | 0.041081 | 0.032476 | 0.066702 |
| Breakout | 0.041256 | 0.013440 | 0.188470 |
| Freeway | 0.010575 | 0.013813 | 0.005960 |
| Seaquest | 0.024695 | 0.008265 | 0.047175 |
| SpaceInvaders | 0.037548 | 0.008840 | 0.102025 |

Measured wall time includes initialization and compilation. PC load varied; these are observed timings, not a controlled speed benchmark. W&B runtime is import time and must not be read as training time.

| Game | QRC minutes | ET minutes |
| --- | --- | --- |
| Asterix | 27.3 | 36.6 |
| Breakout | 29.4 | 34.4 |
| Freeway | 29.5 | 38.2 |
| Seaquest | 31.3 | 36.9 |
| SpaceInvaders | 30.0 | 34.9 |

## What Changed and What Remains Different

| Item | Implemented behavior | Interpretation |
| --- | --- | --- |
| Shared representation | Linear Theta uses Q-head features; ET loss does not train the trunk | No separate deep trace network |
| Algorithm 3 recursion | Full final-layer e^w target; mixed trace enters subsequent QRC updates | ET genuinely influences Q weights |
| Line 15 target | Use e^w, consistent with main-text p.5 and appendix explanation | Already in supplied agent; apparent typo interpretation, not a new fix |
| Paired RNG | Same initial Q/h weights, environment seed and exploration key | Prevents an extra Theta key split from changing the control stream |
| New predictor initialization | Theta starts at zero and then learns | Tested jointly with normalization; effect not isolated |
| New predictor update | Normalize Theta gradients by feature energy | Optimization variant, not demonstrated mechanism accuracy |
| Current action input | Disabled in tested candidate | Predicting all action columns is not conditioning on current action |
| Prediction ordering | Pre-Theta-update prediction in tested candidate | Paper places the mixture after Theta update |
| Base learner | Retains QRC gradient correction; substitutes head-kernel trace only | Adaptation, not exact paper QET reproduction |

Reference: [Expected Eligibility Traces, pp.5 and 15](https://arxiv.org/pdf/2007.01839). The apparent line-15 typo is inferred from internal consistency, not an author-confirmed erratum.

## Progress and Next Decision

| Stage | Change or question | Observed outcome / status |
| --- | --- | --- |
| Earlier research | Eta=1 control and shared-key splitting | Research log records paired-RNG correction; old report retained separately |
| Local implementation checks | Zero initialization and feature-energy normalization | Candidate selected for bounded screening; no proof of expected-trace mechanism |
| 14 Oct: five-game 1M screen | One seed, same rate, matched starts | Higher endpoints in four games; Breakout remains behind |
| 15 Oct: Breakout 3M follow-up | Same source and exploration schedule | Exact first-1M reproduction; endpoint gap remains -3.38 |
| Proposed next ablations | Current-action conditioning, prediction ordering, then both | Not trained in these reported runs; must be tested against unchanged controls |
| Before a broad claim | Held-out seeds, per-method rate tuning, full budgets and damping controls | Still missing; no claim that historical weak seeds are fixed |

The next useful question is whether action conditioning or prediction timing improves returns, rather than merely reducing trace regression loss. Keep R27 and this frozen candidate separate. A short-lambda or zero-Theta control is needed to distinguish expected-trace learning from damping.

## Reproduction and Evidence

[Reviewed code, CSVs and instructions](https://github.com/mubashir20023/eligibility-traces/tree/local-experiment-on-et). The new agent filename is `algorithms/qrc-et-algo3-lab.py`; its flags are explicit in the reproduction instructions. Do not overwrite the cluster's R27 agent.

All 12 imported result histories are complete. The export independently reconciled **16,000 rolling-return records with 216,716 episode records**. Counts include overlapping Breakout prefixes and are audit-work counts, not independent samples. Exported CSVs retain every 1k-step sample. W&B plots receive every 5k-step sample without additional smoothing; scores and diagnostics use the full 1k-step CSVs.

All runs use CPU, seed 601, q_lr=0.002, gamma=0.99 and lambda=0.8. ET uses eta=0.3, Theta lr=0.1, paired RNG, e^w target and the Algorithm-3 lag. Zero initialization and normalization are enabled; action conditioning, warmup and post-update prediction are disabled.

Evidence cutoff: **15 Oct 2026 11:40 Pakistan time (UTC+5)**. Per-run start/finish times, training minutes, initialization/source/data hashes and package versions are preserved in `evidence.json`. No raw returns are averaged across games. No significance test or confidence interval is justified by this single-seed, fixed-rate screen.
