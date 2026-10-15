# QRC and experimental QRC-ET: frozen local comparison

This package contains **QRC and experimental QRC-ET only**. It documents five
1M-step matched pairs and a 3M-step Breakout follow-up, all using seed 601 and
q_lr=0.002. See [REPORT.md](REPORT.md) for the results, implementation differences,
timing, limitations and proposed next experiments. This is not a multi-seed or
5M-step validation. The 3M Breakout run reproduces the first 1M exactly, so those
prefixes are not independent replications.

## Agent and flags

The new file is `algorithms/qrc-et-algo3-lab.py`. It is byte-for-byte identical
to the experimental agent used in the completed local tests:

```text
SHA256 1023f4ed6b63ab91ea8d97df617523be07ada970edde7ee27b029286aec759c6
```

The tested change is zero-initialized Theta plus feature-energy-normalized Theta
gradients. The source defaults keep these features off; **the flags matter**.
The full-vector `e_w` target was already in the supplied Algorithm-3 agent. Its
use is consistent with page 5 and the appendix's explanation of expected
last-layer traces. Treat the conflicting printed line 15 as an apparent typo,
not an author-confirmed erratum.

The tested agent is an adaptation to QRC, not exact paper QET: it retains QRC's
gradient correction, modifies the head kernel trace only, uses a state-only
predictor, and mixes the prediction computed before the current Theta update.
Current-action conditioning and post-update prediction are optional unvalidated
variants in this source, not settings used for these results.

## Copy safely to the cluster

Copy **only the new lab filename** to `algorithms/` in the cluster checkout.
Do not overwrite `algorithms/qrc-et-algo3.py`, alter R27, or reuse existing run
directories. No cluster jobs were submitted by this package.

For example, a fresh Breakout comparison at the reported 3M budget is:

```bash
python algorithms/qrc.py \
  --env_type minatar --env_id MinAtar/Breakout-v1 --net_arch minatar \
  --seed 601 --total_timesteps 3000000 --explore_frac 0.16666666666666666 \
  --q_lr 0.002 --exp_name LOCAL_REPRO_qrc_3M --no-save-model

python algorithms/qrc-et-algo3-lab.py \
  --env_type minatar --env_id MinAtar/Breakout-v1 --net_arch minatar \
  --seed 601 --total_timesteps 3000000 --explore_frac 0.16666666666666666 \
  --q_lr 0.002 --et_eta 0.3 --et_lr 0.1 --et_algo3 --et_paired_rng \
  --et_zero_init --et_normalize_grad \
  --exp_name LOCAL_REPRO_state_nlms_3M --no-save-model
```

For 1M use `--total_timesteps 1000000 --explore_frac 0.5`. A later, separately
approved 5M test should use `--explore_frac 0.1` to preserve the 500k-step schedule.
The ordinary agent entry point and the local harness share the agent update,
but logging/runtime differ; byte-identical logs across machines are not promised.

## Recompute the published measurements

`data/*.csv` retains every 1k-step sample of rolling return, sampled TD squared
error, trace regression loss and training elapsed time. `evidence.json` stores
full-precision metrics, matched initialization hashes, source/data hashes,
package versions and UTC start/finish times. It excludes private filesystem
paths, credentials and correspondence. QRC trace loss is missing, not zero.

```bash
python -m unittest discover -s experiments/qrc_et_local -p 'test_*.py' -v
python experiments/qrc_et_local/publish_wandb.py --github YOUR_REVIEWED_BRANCH_URL
```

With the agent dependencies installed, `python experiments/qrc_et_local/verify_agent.py`
also checks actual JAX updates: eta=1 recovers QRC, learned predictions influence
Q weights, and episode termination clears both trace paths. These short checks
verify implementation behavior, not control performance or expected-trace accuracy.

These commands validate exported data and regenerate the Markdown narrative;
they do not train or publish. The exporter also checked rolling-return records
against raw episode rewards before publication. The raw episode files remain in
the local experiment archive; their hashes are retained in the public evidence.

The recorded runtime was CPU JAX 0.11.2, flax 0.12.10, optax 0.2.8,
numpy 2.5.4, gymnasium 0.29.1 and MinAtar 1.0.15. These are the observed versions,
not a claim that arbitrary newer dependency combinations reproduce the runs.

## Publishing and appending

`publish_wandb.py --push --key-file PATH --github URL` creates the new report
on first use and refreshes **only this new report** on subsequent use through
`publication.json`. Keep that receipt. Existing source/result imports are reused
only after checking their dataset ID, digest and completed-audit marker. The
script never reads an API key from the repository, modifies historical runs,
or changes project privacy. Pass `--sdk-dir` only when using a separately
installed W&B SDK. Report creation needs `wandb` and `wandb-workspaces`.

Do not append new experiments into the frozen CSVs. Give them separate result
identities and explicitly add their budgets, seeds, source hashes and comparisons
in a reviewed revision. Action conditioning and prediction ordering remain
proposed ablations, not completed MinAtar evidence.
