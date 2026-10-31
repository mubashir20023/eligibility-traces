# QRC and Expected Eligibility Traces: Master File and Evaluation Index

**Evidence cutoff:** 23 October 2026  
**Purpose:** One navigation point for experiment protocols, code, results, audits, raw-data locations, reports, and conclusions in this project.

> Local links below are relative to the repository root. In this checkout the root is `C:\Users\Mubashir\stream-rep-rl`. Some older readouts refer to cluster storage (`G:\qrc-et-local-results`) that is not itself a local repository path; where the returned copy exists, its local archive is linked instead. A folder link is a location, not a promise that every raw record is individually previewable in Markdown.

## Bottom Line

There is not yet a robust, general deep-QRC improvement from expected traces. The evidence is mixed and strongly dependent on task and implementation:

- **Promising Freeway result:** R27 Algorithm 3 at `q_lr=0.002`, `eta=0.3` scored 58.37 over 15 seeds versus QRC 51.43 at QRC's earlier best rate `5e-4` (paired difference +6.94, reported interval ±4.19, `p=0.00315`). Ten seeds were fresh; the five-seed rate-selection cohort is included in the 15-seed total. This is a Freeway result, not a multi-game win. The supplied Slurm accounting showed array indices 0–27 completed and 28–29 suspended.
- **Earlier default-rate result:** at `q_lr=1e-4`, the 15-seed Freeway comparison was QRC 47.96 versus Algorithm 3 40.57, ET lower by 7.4 (`p=0.0004` as reported). The higher-rate R27 finding is therefore important but rate-specific.
- **Local five-game screen:** the separate experimental candidate (seed 601, 1M steps) finished higher in four of five games, lower in Breakout. It is `n=1`, so this is a screen only. On the same Breakout seed the deficit widened from -1.16 at 1M to -3.38 at 3M.
- **Mechanism studies:** raw-trace-mean ET reduced tabular noisy-reward RMSE-AUC by 13.49% on 15 held-out seeds, but the recursive Theta variant was 73.06% worse. This tabular result did not transfer to a clear neural learned-trace advantage: in the 15-seed neural confirmation, learned ET was 7.97–10.50% worse than a prediction-disabled/damping control at `eta=.3` under reward noise, with exact greedy-policy returns tied.
- **Most recent local screen:** action-conditioning and prediction-order variants did not beat the damping control consistently. They do not justify an expensive MinAtar run.

**Recommendation:** preserve the positive Freeway result as a task-specific finding, but do not claim general superiority. Ask Peter and Mathieu whether to prioritize (a) an independent fresh-seed Freeway confirmation with a frozen rate, or (b) a more diagnostic task/mechanism experiment that separates prediction quality from trace shortening/damping. Do not launch another broad sweep until that decision is made.

## Terminology and Comparability

- **QRC baseline:** production QRC implementation.
- **R27 Algorithm 3:** the original production QRC-ET code path used by the R27 cluster jobs; keep it distinct from the lab candidate.
- **Experimental ET / lab ET:** zero-initialized Theta plus normalized Theta-gradient candidate used in the local MinAtar comparison. This is an Algorithm-3-style QRC adaptation, not a faithful paper reproduction.
- **Damping / prediction-disabled control:** ET machinery with predicted component disabled; it changes the effective trace and is not automatically a perfectly matched control.
- **RMSE-AUC / value error:** prediction/evaluation diagnostics, not game reward. Lower is better for these errors.
- **Training return:** logged online rolling return, not an independent final-policy evaluation.
- Lower TD error or trace-regression loss alone does not demonstrate better control. Never average raw reward values across different games.

## Results Summary

| Study | Design | Main evaluation | Outcome / status |
|---|---|---|---|
| MinAtar, five games | QRC vs experimental ET; seed 601; 1M steps; `q_lr=.002` | Final rolling-100 return over last 10%; whole-budget curve average | ET higher in Asterix (+.71), Freeway (+1.74), Seaquest (+.36), SpaceInvaders (+1.96); lower in Breakout (-1.16). Single-seed screen. |
| Breakout extension | Same seed 601; QRC vs experimental ET; 1M/2M/3M | Matched-budget last-10% endpoints | QRC 9.67/12.52/14.18 vs ET 8.51/10.51/10.81; ET gap -1.16/-2.01/-3.38. Longer training did not close it. Same seed, not replication. |
| R27 Freeway | Algorithm 3, eta .3, QRC lr .002, 15 seeds | Paired final-window scores | 58.37 vs QRC 51.43 at earlier best QRC rate .0005; +6.94, reported ±4.19, p=.00315. Encouraging, rate-specific. |
| R27 Asterix | Algorithm 3 vs QRC at .002; only 3 pairs in supplied summary | Paired final-window score | 21.18 vs 20.33; +.85, ±6.58, p=.634. Inconclusive. |
| Gridworld stochasticity pilot | Original lagged ET; 3 seeds × slip 0/.2/.5 × eta/rate grid | Final-window absolute error vs exact V* | Strong rate dependence; QRC can diverge at high rate while ET remains finite. Pilot only; two nonfinite records retained, no broad causal claim. |
| Gridworld stochasticity confirmation | Original ET; 15 seeds; slip 0/.5; eta 0 | Final-window V error | Noise-by-method interaction passed (`p=.00757`), but at slip .5 ET vs QRC and ET vs disabled control both inconclusive after correction. Full prespecified success criteria not met. |
| Tabular trace-rate test | 15 held-out seeds, delayed reward | Whole-budget state RMSE-AUC | Ordinary-trace-mean ET 13.49% lower error than QRC, 15/15 seeds; recursive Theta ET 73.06% higher error. Task-specific. |
| Predictor-rate test | Fresh 15-seed cluster cohort | Noisy reward RMSE-AUC | Ordinary target fixed gain: 20.77% lower error; count gain: 23.63% lower error vs selected QRC. Count gain beat fixed gain 14/15. Rates touch grid boundary. |
| Target × gain ablation | 5-seed tabular follow-up | Noisy-reward RMSE-AUC | Ordinary target + count gain best in this screen; target and gain effects are not a clean standalone generalization. |
| Neural-MDP, first screen | 5 seeds; 6 noise cells; 8,192 updates | Value RMSE-AUC | Learned ET worse than damping under noisy terminal reward; policy returns tied. Exploratory. |
| Neural-MDP cluster confirmation | 15 fresh seeds; 810 runs; 26.54M updates | Value RMSE-AUC and exact greedy returns | At eta .3/noisy reward, learned ET 7.97%, 11.70%, 10.50% higher error than damping for transition flips 0/.2/.5. Across eta .0/.3/.6 it was 5.19–13.66% worse in noisy cells. Returns tied. |
| Neural-MDP action/timing | 5 seeds; 6 cells; 6 variants; 180 runs | Value RMSE-AUC | No consistent improvement from action conditioning or post-update ordering; noisy cells favor damping. No variant promoted. |
| Online oracle evaluation | 5 seeds; six noise cells | Value RMSE-AUC | Oracle improves over learned ET, but is nearly tied with ordinary evaluator; does not demonstrate a practical ET advantage over QRC. |
| Frozen expected-trace update | Frozen model banks; 8,100 cases | One-step update variance | At eta .3 and reward noise, variance reductions 1.98%, 3.47%, 4.29% as transition flip rises. Mechanism, not online return evidence. |
| Theta gain screen | Small local neural-MDP grid | Prediction/value diagnostics | Lowering gain did not close learned-ET gap to damping under noise. Not a game result. |
| H health / early checkpoint | Replays of existing five seeds | H norms, update decomposition | H learns by steps 4–8, then shrinks toward float32 zero around 2.86k updates under the tested regularization. Diagnostic only; not proven cause of game gap. |
| H frozen-update review | 120 saved banks; 6,480 exact cases | Variance decomposition | At update 128, H covariance changes oracle variance effect by at most 0.00015 percentage points in baseline-QRC banks. Does not rule out earlier effects. |
| Predictor diagnosis / merging paths / fixed updates | Bounded model and replay audits | Target error, moment equations, source/data checks | Supporting diagnostics; read the linked reports for exact scope. They do not establish a deep-control win. |

## Detailed Evaluations

### 1. MinAtar and Breakout

Five-game seed-601 final-window returns and curve averages:

| Game | QRC score | Experimental ET score | ET−QRC | QRC / ET curve avg. |
|---|---:|---:|---:|---:|
| Asterix | 3.44 | 4.15 | +0.71 | 1.692 / 2.048 |
| Breakout | 9.67 | 8.51 | -1.16 | 5.135 / 4.719 |
| Freeway | 22.08 | 23.82 | +1.74 | 9.429 / 9.471 |
| Seaquest | 0.15 | 0.51 | +0.36 | 0.252 / 0.444 |
| SpaceInvaders | 23.71 | 25.66 | +1.96 | 14.832 / 14.951 |

Seed 601, `q_lr=.002`, matched initialization/exploration keys. Score window is `900k < step <= 1M`. These are training returns. Individual runtime, sampled TD-squared, and trace losses are in the report table and per-game HTML files linked below. QRC has no learned trace-loss metric; this is N/A, not zero.

Breakout matched-budget endpoint series: QRC **9.67 → 12.52 → 14.18**; experimental ET **8.51 → 10.51 → 10.81**, at 1M/2M/3M. At 3M, curve average was 10.153 vs 8.283; final sampled TD-squared was .052241 vs .005895, and ET trace loss was .173820. The lower ET TD diagnostic did not yield the higher return. First-million log fields matched between the 1M and 3M same-seed runs; this establishes reproducibility of the prefix, not independent replication.

### 2. R27 Cluster Summary

User-provided `summarize_27.py` output (the exact script is not part of this checkout) reported:

| Freeway cohort / comparator | Algorithm 3 | QRC | Difference | Reported inference |
|---|---:|---:|---:|---|
| Seeds 1,2,3,42,999; same lr .002 | 59.17 | 54.24 | +4.93 | ±6.08; p=.0876; rate-selection cohort |
| Same five; QRC earlier best lr .0005 | 59.17 | 53.95 | +5.22 | ±8.23; p=.153; exploratory |
| Fresh seeds 4–13; same lr .002 | 57.97 | 39.70 | +18.28 | ±11.03; p=.00455 |
| Fresh seeds 4–13; QRC earlier best .0005 | 57.97 | 50.17 | +7.80 | ±5.85; p=.0145 |
| All 15; same lr .002 | 58.37 | 44.54 | +13.83 | ±7.87; p=.00208 |
| All 15; QRC earlier best .0005 | 58.37 | 51.43 | +6.94 | ±4.19; p=.00315 |
| All 15; QRC default .0001 | 58.37 | 47.96 | +10.42 | ±3.26; p=7.95e-06 |

The first five seeds were involved in choosing the ET rate; fresh seeds 4–13 provide the less biased comparison. Scheduler accounting supplied by the user marked indices 0–27 `COMPLETED`, 28–29 `SUSPENDED`; therefore the supplied evidence does not support saying all 30 array tasks completed. R27 remains a different code path from the experimental local candidate.

### 3. Stochasticity, Eta, and Trace Construction

- **Original ET stochasticity pilot:** 135 planned combinations, 20k updates, seeds 701–703, slip 0/.2/.5, eta 0/.3/.6/1 and Q rates .01/.03/.1. All eta=1 identities passed; two nonfinite records were retained (QRC and eta=1 ET, seed 701, rate .1, slip 0). At rate .03, eta=0 could be worse at slip 0 but better at slip .5; at rate .1 ET was more stable than QRC. This is a pilot on three seeds and does not isolate learned prediction from trace shortening.
- **15-seed confirmation:** original lagged Algorithm-3 ET, seeds 17001–17015, slip 0/.5, eta 0. Prespecified noise interaction was -0.38739, 95% CI [-.65393,-.12085], `p=.00757`. At slip .5, ET−QRC was -.10222 (Holm p=.76935) and ET−prediction-disabled ET was +.01074 (Holm p=.83406). Thus interaction direction replicated, but the two key pairwise effects were inconclusive; the full success criteria were not met.
- **Zero-initialization pilot/confirmation:** separate five-seed reports exist. Consult the report files; these results concern a bounded gridworld setup, not the five-game benchmark.
- **Trace-rate calibration and held-out test:** selected rates were sealed before fresh evaluation. In the held-out 15-seed noisy-reward test, QRC RMSE-AUC .187105; oracle .156981 (-16.10%); recursive Theta .323801 (+73.06%); ordinary raw-trace-mean .161870 (-13.49%, 15/15 wins). Deterministic reward gave ordinary-trace-mean .144861 vs QRC .139642, 3.74% worse. Selected rates touch the upper grid boundary. A later predictor-rate test found fixed-gain ordinary target .149623 (-20.77%) and count-gain .144224 (-23.63%) vs selected QRC .195820 in the noisy cell; count beat fixed gain 14/15. This is value prediction on a tabular task, not evidence of game-return gains.
- **Target × gain:** five-seed follow-up varied recursive vs ordinary target and fixed vs count gain. On noisy reward, RMSE-AUCs were .321833 (recursive/fixed), .183362 (recursive/count), .306144 (ordinary/fixed), .158431 (ordinary/count); QRC .185623, oracle .153726. This is exploratory and both design factors matter.
- **Oracle evaluation:** in a fixed-policy evaluation, oracle trace largely removed the learned ET error gap but was effectively tied with ordinary evaluation. This does not demonstrate useful control improvement; the oracle is privileged.
- **Frozen update moments:** exact expected traces reduced update variance by 1.98%, 3.47%, and 4.29% for transition flip probabilities 0/.2/.5 at eta .3 and noisy reward. The prediction-disabled control reduces variance more but shifts the conditional mean. Variance alone is not efficacy.

### 4. Neural-MDP and Predictor Diagnostics

- **Initial screen:** five seeds × six transition/reward-noise cells, 8,192 steps. At eta .3 with noisy terminal reward, learned ET's RMSE-AUC was 3.08–10.80% higher than prediction-disabled damping; greedy returns tied.
- **Fresh 15-seed confirmation:** 810 runs / 26,542,080 updates, all finite and independently audited. At eta .3/noisy reward, learned ET had 7.97%, 11.70%, and 10.50% higher RMSE-AUC than damping for transition flips 0/.2/.5. Across eta .0/.3/.6, it was 5.19–13.66% worse in noisy-reward cells. Greedy returns tied. Learned predictor loss could be lower even when value error was worse.
- **Action conditioning × prediction timing (23 Oct):** 180 runs, five fresh seeds, six noise cells, six methods. All finite; independent audit passed. No consistent benefit from current-action conditioning or post-update prediction. In all three noisy-reward cells, every learned variant had higher mean RMSE-AUC than damping. Exact greedy returns tied, so this task did not discriminate control outcomes.
- **Online oracle screen:** five seeds × six cells. At noisy reward, oracle ET error AUC was 7.87–11.10% higher than damping and lost on all five seed pairs in each cell; the prespecified promotion gate failed. This oracle is not a deployable learner.
- **H-head activity:** replayed existing seeds show H begins learning by updates 4–8, then becomes exactly zero around updates 2,861–2,895 under tested regularization. This is consistent with regularization-dominated shrinkage/float32 underflow, not proof it caused the MinAtar deficit.
- **Early frozen H review:** at update 128, oracle variance reduction was 2.44%, 5.27%, 6.15% for flip 0/.2/.5; in baseline-QRC banks H-correction covariance altered the effect by at most .00015 percentage points. This does not rule out earlier interactions.
- **Fixed-update, predictor-diagnosis, merging-path, and Theta-gain studies:** support the detailed reports linked in the file map. These are mechanism/error diagnostics and do not establish a better policy.

## File Map: Reports, Protocols, Code, Results, Audits

### Master reports

- [Consolidated test-results DOCX (23 Oct)](QRC_ET_All_Tests_and_Results_2026-10-23.docx) — narrative tables and earlier evidence cutoff.
- [DOCX generator source](QRC_ET_All_Tests_and_Results_2026-10-23.py).
- [Local two-method MinAtar report](qrc_et_local/REPORT.md) — full per-game/per-seed table, costs, Breakout progression and interpretation.
- [Local report README](qrc_et_local/README.md), [evidence/provenance](qrc_et_local/evidence.json), [W&B publication metadata](qrc_et_local/publication.json), [W&B preparation script](qrc_et_local/prepare_publication.py), [W&B publisher](qrc_et_local/publish_wandb.py).
- [W&B: historical QRC and Expected Traces report](https://forge.coreweave.com/wandb/mubbashirahmed/qrc-et-algo3-minatar/reports/QRC-and-Expected-Traces:-Research-Progress--VmlldzoxODEwMTk1Ng==).
- [W&B: QRC vs Experimental QRC-ET report](https://forge.coreweave.com/wandb/mubbashirahmed/qrc-et-algo3-minatar/reports/QRC-vs-Experimental-QRC-ET-%7C-Five-Game-Screen-and-Breakout-Follow-up--VmlldzoxODExNTc2Nw==).
- [GitHub project](https://github.com/mubashir20023/eligibility-traces/tree/local-experiment-on-et) — branch URL was present in the report; verify branch availability before sharing externally.

### Experiment folders and key files

Each row links the protocol/readout and names the folder holding the full run-specific artifacts (plans, statuses, JSON/CSV tables, validators, plots and raw arrays as applicable).

| Experiment | Protocol/source | Readout / audit | Full artifact location |
|---|---|---|---|
| Local MinAtar QRC vs experimental ET | [README](qrc_et_local/README.md), [table generator](qrc_et_local/report_tables.py) | [REPORT](qrc_et_local/REPORT.md), [evidence](qrc_et_local/evidence.json), [layout audit](qrc_et_local/TABLE_LAYOUT_AUDIT.json) | [`experiments/qrc_et_local/`](qrc_et_local/) — per-game CSVs under `data/`; individual and combined report tables under `tables/` |
| Five-game local action/timing follow-up | [Protocol](qrc_et_action_timing/PROTOCOL.md), [runner](qrc_et_action_timing/run.py), [tests](qrc_et_action_timing/test_action_timing.py), [reviewer](qrc_et_action_timing/review.py) | [READOUT](qrc_et_action_timing/local_20261023/READOUT.md), [comparison](qrc_et_action_timing/local_20261023/comparison.json), [validation](qrc_et_action_timing/local_20261023/VALIDATION.json), [plan](qrc_et_action_timing/local_20261023/plan.json) | [`experiments/qrc_et_action_timing/local_20261023/`](qrc_et_action_timing/local_20261023/) — `raw_000.npz`–`raw_035.npz` and sealed source snapshots |
| Neural-MDP local screen and 15-seed cluster confirmation | [Protocol](qrc_et_neural_noise/PROTOCOL.md), [runner](qrc_et_neural_noise/run.py), [agent](qrc_et_neural_noise/agent.py), [task](qrc_et_neural_noise/task.py), [cluster SLURM](qrc_et_neural_noise/run.slurm), [package script](qrc_et_neural_noise/package.py) | [local readout](qrc_et_neural_noise/local_20261021/READOUT.md), [cluster review](qrc_et_neural_noise_review/cluster_return_20261021/READOUT.md), [review summary](qrc_et_neural_noise_review/cluster_return_20261021/SUMMARY.md), [verified metrics](qrc_et_neural_noise_review/cluster_return_20261021/VERIFIED_RESULTS.json), [validation](qrc_et_neural_noise_review/cluster_return_20261021/VALIDATION.json) | [`qrc_et_neural_noise/local_20261021/`](qrc_et_neural_noise/local_20261021/), [`qrc_et_neural_noise_review/cluster_return_20261021/`](qrc_et_neural_noise_review/cluster_return_20261021/), and [returned cluster bundle](ET_neural_noise_cluster_20261021.tgz) |
| Neural action-conditioning / timing screen | [Protocol](qrc_et_action_timing/PROTOCOL.md) and linked runner above | [READOUT](qrc_et_action_timing/local_20261023/READOUT.md), [VALIDATION](qrc_et_action_timing/local_20261023/VALIDATION.json) | [`qrc_et_action_timing/local_20261023/`](qrc_et_action_timing/local_20261023/) |
| Tabular trace-rate calibration | [Protocol](qrc_et_stochasticity/TRACE_RATE_CALIBRATION.md) | [calibration readout](qrc_et_stochasticity/trace_rate_calibration_20261020/READOUT.md), [validation](qrc_et_stochasticity/trace_rate_calibration_20261020/VALIDATION.json) | [`qrc_et_stochasticity/trace_rate_calibration_20261020/`](qrc_et_stochasticity/trace_rate_calibration_20261020/) |
| Fresh 15-seed tabular trace-rate return | Same protocol above | [independent audit readout](qrc_et_stochasticity/trace_rate_cluster_return_20261020/READOUT.md), [primary analysis](qrc_et_stochasticity/trace_rate_cluster_return_20261020/PRIMARY_ANALYSIS.json), [audit receipt](qrc_et_stochasticity/trace_rate_cluster_return_20261020/AUDIT_RECEIPT.json) | [`qrc_et_stochasticity/trace_rate_cluster_return_20261020/`](qrc_et_stochasticity/trace_rate_cluster_return_20261020/) — returned source/results retained under `received/` |
| Fresh predictor-rate test | [Protocol](qrc_et_stochasticity/PREDICTOR_RATE_PROTOCOL.md) | [cluster-return readout](qrc_et_stochasticity/predictor_rates_cluster_return_20261021/READOUT.md), [primary analysis](qrc_et_stochasticity/predictor_rates_cluster_return_20261021/PRIMARY_ANALYSIS.json), [summary](qrc_et_stochasticity/predictor_rates_cluster_return_20261021/SUMMARY.md), [validation](qrc_et_stochasticity/predictor_rates_cluster_return_20261021/FINAL_CHECK.json) | [`qrc_et_stochasticity/predictor_rates_cluster_return_20261021/`](qrc_et_stochasticity/predictor_rates_cluster_return_20261021/) |
| Predictor-rate initial submission / I/O recovery | [Launch instructions](qrc_et_stochasticity/predictor_rates_20261021_io_retry/LAUNCH.md) | [readout](qrc_et_stochasticity/predictor_rates_20261021_io_retry/READOUT.md), [recovery receipt](qrc_et_stochasticity/predictor_rates_20261021_io_retry/IO_RECOVERY_COMPLETED.json), [technical abort](qrc_et_stochasticity/predictor_rates_20261021/TECHNICAL_ABORT.json) | [`qrc_et_stochasticity/predictor_rates_20261021_io_retry/`](qrc_et_stochasticity/predictor_rates_20261021_io_retry/) and [`predictor_rates_20261021/`](qrc_et_stochasticity/predictor_rates_20261021/) |
| Target × gain ablation | [Protocol](qrc_et_stochasticity/TRACE_TARGET_GAIN_ABLATION.md) | [readout](qrc_et_stochasticity/trace_target_gain_20261021/READOUT.md), [results](qrc_et_stochasticity/trace_target_gain_20261021/RESULTS.md), [validation](qrc_et_stochasticity/trace_target_gain_20261021/VALIDATION.json) | [`qrc_et_stochasticity/trace_target_gain_20261021/`](qrc_et_stochasticity/trace_target_gain_20261021/) |
| Original-ET stochasticity pilot | [README](qrc_et_stochasticity/README.md) | [completed pilot](qrc_et_stochasticity/COMPLETED_PILOT.md), [protocol](qrc_et_stochasticity/cluster_eta_noise_protocol.md) | [`qrc_et_stochasticity/`](qrc_et_stochasticity/) — pilot outputs are on cluster path `G:\qrc-et-local-results\gridworld_stoch_eta_pilot_20261019` unless copied into a named local subfolder |
| Original-ET 15-seed stochasticity confirmation | [Confirmation plan](qrc_et_stochasticity/CONFIRMATION_PLAN.json) | [completion report](qrc_et_stochasticity/COMPLETED_CONFIRMATION.md), [confirmation summary](qrc_et_stochasticity/CONFIRMATION.md) | [`qrc_et_stochasticity/`](qrc_et_stochasticity/) — cluster path `G:\qrc-et-local-results\gridworld_stoch_eta_confirmation15_20261019`; per-seed artifacts listed in report |
| Oracle evaluation | [Protocol](qrc_et_stochasticity/ORACLE_EVALUATION.md) | [readout](qrc_et_stochasticity/oracle_evaluation_5seed_20261020/READOUT.md), [results](qrc_et_stochasticity/oracle_evaluation_5seed_20261020/RESULTS.md), [analysis](qrc_et_stochasticity/oracle_evaluation_5seed_20261020/analysis.json), [email draft](qrc_et_stochasticity/oracle_evaluation_5seed_20261020/EMAIL_DRAFT.md) | [`qrc_et_stochasticity/oracle_evaluation_5seed_20261020/`](qrc_et_stochasticity/oracle_evaluation_5seed_20261020/) |
| Frozen update / merging paths | [Merging protocol](qrc_et_stochasticity/MERGING_PATHS_PROTOCOL.md) | [frozen moments report](qrc_et_stochasticity/merging_paths_20261020/READOUT.md), [results](qrc_et_stochasticity/merging_paths_20261020/RESULTS.md), [merge diagnosis](qrc_et_stochasticity/merging_diagnosis_20261020/READOUT.md) | [`merging_paths_20261020/`](qrc_et_stochasticity/merging_paths_20261020/) and [`merging_diagnosis_20261020/`](qrc_et_stochasticity/merging_diagnosis_20261020/) |
| Trace seed diagnostics / replay | [Replay overview](qrc_et_stochasticity/REPLAY.md) | [replay findings](qrc_et_stochasticity/replay_20261019/FINDINGS.md), [seed trace review](qrc_et_stochasticity/seed_trace_review_20261020/READOUT.md), [seed audit](qrc_et_stochasticity/seed_trace_review_20261020/AUDIT.json) | [`replay_20261019/`](qrc_et_stochasticity/replay_20261019/), [`seed_diagnostics_20261020/`](qrc_et_stochasticity/seed_diagnostics_20261020/), [`seed_trace_review_20261020/`](qrc_et_stochasticity/seed_trace_review_20261020/) |
| Zero-initialization confirmation | [Protocol/report](qrc_et_stochasticity/ZERO_INIT_CONFIRMATION.md) | [pilot](qrc_et_stochasticity/ZERO_INIT_PILOT.md), [completed pilot](qrc_et_stochasticity/COMPLETED_PILOT.md), [confirmation](qrc_et_stochasticity/COMPLETED_CONFIRMATION.md) | [`zero_init_5seed_20261019/`](qrc_et_stochasticity/zero_init_5seed_20261019/) plus per-study files linked above |
| Theta-gain screen and recovery | [Screen protocol](qrc_et_theta_gain_screen/PROTOCOL.md) | [recovery report](qrc_et_theta_gain_recovery/RECOVERY.md), [review summary](qrc_et_theta_gain_review/SUMMARY.md), [active-H review](qrc_et_active_h_update_review/SUMMARY.md) | [`qrc_et_theta_gain_screen/local_20261021/`](qrc_et_theta_gain_screen/local_20261021/) and [`qrc_et_theta_gain_recovery/`](qrc_et_theta_gain_recovery/) |
| H health and active-H updates | [H-health protocol](qrc_et_h_health/PROTOCOL.md), [active-H protocol](qrc_et_active_h_update/PROTOCOL.md) | [H-health review](qrc_et_h_health_review/SUMMARY.md), [active-H frozen update](qrc_et_active_h_update_review/SUMMARY.md), [early checkpoint protocol](qrc_et_early_h_checkpoint/PROTOCOL.md) | [`qrc_et_h_health/`](qrc_et_h_health/), [`qrc_et_active_h_update/`](qrc_et_active_h_update/), [`qrc_et_early_h_checkpoint/`](qrc_et_early_h_checkpoint/) — raw/replay archives may be cluster-only as described in summaries |
| Early window and fixed update | [Early-window protocol](qrc_et_early_window/PROTOCOL.md), [fixed-update protocol](qrc_et_fixed_update/PROTOCOL.md) | [fixed-update review](qrc_et_fixed_update_review/SUMMARY.md), [online-oracle review](qrc_et_online_oracle_review/SUMMARY.md) | [`qrc_et_early_window/`](qrc_et_early_window/), [`qrc_et_fixed_update/`](qrc_et_fixed_update/), [`qrc_et_online_oracle/`](qrc_et_online_oracle/) |
| Predictor diagnosis | [Protocol](qrc_et_predictor_diagnosis/PROTOCOL.md) | [readout](qrc_et_predictor_diagnosis/local_20261021/READOUT.md), [summary](qrc_et_predictor_diagnosis/local_20261021/SUMMARY.md), [validation](qrc_et_predictor_diagnosis/local_20261021/VALIDATION.json) | [`qrc_et_predictor_diagnosis/local_20261021/`](qrc_et_predictor_diagnosis/local_20261021/) |

### Cluster run locations and handoff artifacts

The following paths are mentioned in the saved experiment records and user-provided cluster outputs. They are **cluster paths**, not guaranteed to exist on this PC:

- R27 project outputs: `/fred/oz511/muahmed/code/stream-rep-rl/logs/` and `R27_report_results.tgz`. The downloaded archive was reported as `C:\Users\Mubashir\Downloads\R27_report_results.tgz`; it is not present in this repository tree unless separately copied here.
- Stochasticity confirmation: `G:\qrc-et-local-results\gridworld_stoch_eta_confirmation15_20261019`.
- Neural-noise returned results: locally preserved review under [`qrc_et_neural_noise_review/cluster_return_20261021/`](qrc_et_neural_noise_review/cluster_return_20261021/); original package [`ET_neural_noise_cluster_20261021.tgz`](ET_neural_noise_cluster_20261021.tgz).
- The cluster raw stdout/stderr and Slurm accounting are not all included in the repository. A successful Python validation is not by itself scheduler-accounting proof.

### Core production and experimental source files

- [Production Algorithm 3 QRC-ET](../algorithms/qrc-et-algo3.py) — R27/original code path; do not replace it with the lab candidate.
- [Production QRC baseline](../algorithms/qrc.py).
- [Experimental lab agent snapshot](qrc_et_local/README.md) — the MinAtar experiment used a distinct lab copy; the report documents exact options and source hashes. It should not be confused with the production file above.
- [Neural-MDP candidate implementation](qrc_et_neural_noise/agent.py).
- [Action/timing experiment runner](qrc_et_action_timing/run.py).
- [Project README](../README.md).

## Reproducibility and Caveats

1. Use each study's own `plan.json`, `manifest.json`, source snapshot, package/version record, and `VALIDATION.json` as the provenance chain. Do not combine raw records from different study folders unless the readout explicitly defines a matched comparison.
2. Fresh-seed confirmation means seeds were held out from that study's selection cohort; it does not mean the algorithm/hyperparameters were globally tuned or independently replicated across labs.
3. Report matched budgets and matched endpoint windows. The 1M/2M/3M Breakout numbers use the same seed, so they describe one learning trajectory under reproducible prefixes.
4. Keep outputs from local experimental ET, production Algorithm 3, tabular raw-mean ET, oracle ET, and prediction-disabled damping separate. They are not interchangeable implementations.
5. For missing raw files, use the cluster location and transfer/package instructions in the corresponding protocol. Do not silently treat absent files or suspended jobs as completed or as zero results.
6. Local document links refer to files in this repository. For a web-hosted file, use the W&B/GitHub URLs above. Local links are not public share links.

## Suggested Reading Order

1. This index.
2. [Consolidated historical all-tests DOCX](QRC_ET_All_Tests_and_Results_2026-10-23.docx).
3. [R27 versus experimental ET MinAtar report](qrc_et_local/REPORT.md).
4. [15-seed neural-MDP confirmation](qrc_et_neural_noise_review/cluster_return_20261021/READOUT.md).
5. [Held-out tabular trace-rate confirmation](qrc_et_stochasticity/trace_rate_cluster_return_20261020/READOUT.md).
6. [Most recent action/timing factorial screen](qrc_et_action_timing/local_20261023/READOUT.md).
7. [Original-ET stochasticity confirmation](qrc_et_stochasticity/COMPLETED_CONFIRMATION.md).

