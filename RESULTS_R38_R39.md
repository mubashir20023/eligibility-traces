# Non-convolutional networks and pre-trained representations: R38, R39, R39b

Follow-up to `RESULTS_R36_R37.md`, on Peter's two suggestions for finding out why QRC-ET does not improve on QRC on MinAtar.
All QRC-ET runs use the split form (`--et_split --no-et_algo3_lag`, η=0.3). γ = 0.99 everywhere.

New agent options (`apply_r38_r39_patch.py`): `--net_arch mlp` (flatten → Dense 128 → Q-head), `--init_from <model>`
(copy every layer except the Q-head, for the Q and h networks), `--freeze_body` (zero the updates of those layers).

## R39b: pre-trained conv layers, frozen, identical settings (`slurm/R39b_frozen_matched.sh`, 800 runs)

Conv layers from the best R37 QRC run or the best R37 QRC-ET run (`r39_sources.txt`), fresh Q-head, cleared traces / Θ /
optimiser state, conv layers frozen. Both methods: learning rate {2.5e-4, 5e-4, 1e-3, 2e-3} × λ {0.24, 0.8}; QRC-ET Θ/Q
ratio 100. 5 seeds (501-505), 5M steps. Summary: `python summarize_r39b.py`.

| Game | Features from | Best QRC | Best QRC-ET | ET significantly better / worse (of 8 settings) |
|---|---|---|---|---|
| Asterix | QRC | 44.1 | 43.5 | 0 / 0 |
| Asterix | QRC-ET | 40.8 | 40.3 | 0 / 2 |
| Breakout | QRC | 15.2 | 13.9 | 0 / 0 |
| Breakout | QRC-ET | 8.0 | 9.7 | 0 / 0 |
| Freeway | QRC | 60.1 | 59.9 | 0 / 2 |
| Freeway | QRC-ET | 59.7 | 59.6 | 0 / 0 |
| Seaquest | QRC | 16.9 | 7.0 | 1 / 1 |
| Seaquest | QRC-ET | 60.6 | 52.9 | 0 / 0 |
| SpaceInvaders | QRC | 167.7 | 168.9 | 1 / 2 |
| SpaceInvaders | QRC-ET | 142.9 | 134.8 | 1 / 3 |

Over all 80 matched settings QRC-ET is significantly better in 3 and worse in 10. Its wins come from settings where QRC
collapsed or neither method learned (e.g. SpaceInvaders λ=0.8, lr 1e-3); at the settings that work well (λ=0.24, low
learning rate) differences are small and slightly favour QRC. λ=0.8 on frozen features often fails for both.

## R39: pre-trained conv layers with each method's own R37 settings (`slurm/R39_transfer.sh`, 400 runs)

Same transfer, frozen and unfrozen, 10 seeds (501-510), each method at its own tuned R37 learning rate and λ.
ET − QRC final score (bold = significant):

| Game | QRC's features, frozen | QRC's, unfrozen | QRC-ET's features, frozen | QRC-ET's, unfrozen |
|---|---|---|---|---|
| Asterix | **+2.5** | **+10.7** | **+4.7** | **+9.3** |
| Breakout | −1.2 | **−8.0** | +0.4 | −0.7 |
| Freeway | **−1.0** | **−1.6** | −0.4 | **−1.3** |
| Seaquest | +0.1 | **+3.4** | −0.3 | −0.3 |
| SpaceInvaders | **−28.5** | −10.5 | **−14.1** | **−19.1** |

Confounded by the learning rate (Asterix: QRC 1e-3 vs QRC-ET 5e-4; Breakout: QRC 5e-4 vs QRC-ET 2e-3). With identical
settings (R39b) the large Asterix gap disappears (best 44.1 vs 43.5).

## R38: networks without convolution (`slurm/R38_nonconv_tune.sh`, `slurm/R38_nonconv_final.sh`)

Equal-budget tuning as R36 (24 settings per method, seeds 301-303), then each method's best setting on 10 fresh seeds (401-410).
Summaries: `python summarize_r36.py R38mlp` / `R38lin2`, `python summarize_r34c.py R38mlpF` / `R38lin2F`.

**MLP** (720 + 100 runs):

| Game | Best QRC | Best QRC-ET | ET − QRC (95% CI) | ET higher on |
|---|---|---|---|---|
| Asterix | 9.3 | 5.6 | −3.7 [−9.0, +1.6] | 2/10 |
| Breakout | 11.7 | 12.0 | +0.3 [−2.0, +2.7] | 5/10 |
| Freeway | 37.8 | 16.3 | **−21.5 [−37.6, −5.4]** | 3/10 |
| Seaquest | 44.1 | 34.8 | −9.3 [−21.9, +3.4] | 3/10 |
| SpaceInvaders | 110.9 | 98.0 | **−12.9 [−24.1, −1.8]** | 3/10 |

On Freeway 6 of 10 QRC-ET seeds (best setting λ=0.8) collapse to a return of about 0; the 4 that learn score 36-46.
Divergence at λ=0.8 (the λ both methods had): QRC 7/60, QRC-ET 14/180; QRC's other divergences are at λ ≥ 0.9.

**Linear on raw 0/1 pixels** (prefix R38lin2, learning rates 2.5e-3 to 2e-2; 720 + 100 runs). The first attempt (R38lin)
used observation normalisation and diverged almost everywhere: rare pixels become very large inputs and a linear network
has no normalising layer.

| Game | Best QRC | Best QRC-ET | ET − QRC (95% CI) | ET higher on |
|---|---|---|---|---|
| Asterix | 6.6 | 4.3 | **−2.3 [−4.4, −0.2]** | 2/10 |
| Breakout | 3.0 | 2.6 | −0.4 [−1.1, +0.2] | 3/10 |
| Freeway | 18.3 | 20.3 | +2.0 [−2.2, +6.2] | 6/10 |
| Seaquest | 1.0 | 0.9 | −0.1 [−0.3, +0.2] | 4/10 |
| SpaceInvaders | 20.5 | 19.3 | −1.2 [−5.6, +3.1] | 5/10 |

## Conclusions

| Setting | QRC-ET vs QRC |
|---|---|
| One-hot multi-chain (R33-R35) | Better: 13-32% lower value error, every seed |
| MinAtar conv, equal tuning (R36-R37) | Level on all 5 games |
| MinAtar MLP (R38) | Worse on Freeway (seeds collapse) and SpaceInvaders, level elsewhere |
| MinAtar linear, raw pixels (R38) | Level (slightly worse on Asterix) |
| Pre-trained conv, frozen, identical settings (R39b) | Level: better in 3, worse in 10 of 80 settings |

1. With the representation fixed and identical settings, QRC-ET does not help the policy learning itself.
2. Removing convolution does not create an advantage either: an MLP or a linear network on pixels still shares weights
   across many states. The expected-trace gain appears only with one-hot features, where every state has its own feature.
