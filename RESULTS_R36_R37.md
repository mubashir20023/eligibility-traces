# Equal-budget MinAtar tuning: R36-R37

Follow-up to `RESULTS_R33_R35.md`, after Peter's suggestion: is there a MinAtar setting where QRC-ET does better
(for example a different ratio between Θ's and Q's learning rates), with the same amount of tuning for both methods?

## Design

| Method | Learning rates | λ | Θ / Q learning-rate ratio | Settings |
|---|---|---|---|---|
| QRC | 2.5e-4, 5e-4, 1e-3, 2e-3 | 0, 0.24, 0.5, 0.8, 0.9, 0.95 | – | 24 |
| QRC-ET (`--et_split --no-et_algo3_lag`, η=0.3) | 2.5e-4, 5e-4, 1e-3, 2e-3 | 0.24, 0.8 | 30, 100, 300 | 24 |

- **R36** (`slurm/R36_minatar_tune.sh`, job 19651125): all five games, 3 tuning seeds (301-303), 5M steps, 720 runs.
  Θ's learning rate = ratio × Q's (the earlier default, Θ lr 0.1 at Q lr 1e-3, is ratio 100).
- **R37** (`slurm/R37_minatar_final.sh`, job 19802608): each method's best R36 setting per game (`r36_best.txt`),
  10 fresh seeds (401-410), 100 runs.

Score: rolling-100 return over the last 10% of 5M steps. Summaries: `python summarize_r36.py`, `python summarize_r34c.py R37`.

## Final comparison (R37, 10 fresh seeds)

| Game | Best QRC (setting) | Best QRC-ET (setting) | ET − QRC (95% CI) | ET higher on |
|---|---|---|---|---|
| Asterix | 31.1 ± 1.6 (λ=0.5, lr 1e-3) | 29.9 ± 1.2 (λ=0.24, lr 5e-4, ratio 100) | -1.2 [-5.4, +3.0] | 5/10 |
| Breakout | 11.0 ± 0.8 (λ=0.5, lr 5e-4) | 12.6 ± 1.2 (λ=0.8, lr 2e-3, ratio 30) | +1.6 [-1.3, +4.5] | 6/10 |
| Freeway | 56.0 ± 1.2 (λ=0.24, lr 2e-3) | 57.0 ± 0.7 (λ=0.24, lr 2e-3, ratio 30) | +0.9 [-2.2, +4.1] | 4/10 |
| Seaquest | 54.1 ± 4.1 (λ=0, lr 5e-4) | 52.4 ± 4.3 (λ=0.24, lr 5e-4, ratio 100) | -1.6 [-12.3, +9.0] | 3/10 |
| SpaceInvaders | 132.5 ± 6.6 (λ=0.5, lr 5e-4) | 122.7 ± 4.5 (λ=0.24, lr 5e-4, ratio 30) | -9.8 [-26.6, +7.0] | 3/10 |

No significant difference on any game; neither method diverged with its best setting.

## Tuning (R36, 3 seeds)

| Game | Best QRC | Best QRC-ET | Diverged runs QRC / QRC-ET (of 72 each) |
|---|---|---|---|
| Asterix | 31.4 | 32.7 | 8 / 0 |
| Breakout | 18.0 | 15.9 | 0 / 0 |
| Freeway | 58.3 | 58.4 | 8 / 0 |
| Seaquest | 66.0 | 67.8 | 7 / 0 |
| SpaceInvaders | 146.3 | 140.4 | 0 / 0 |

QRC-ET by Θ/Q ratio (best / median score over learning rates and λ):

| Game | ratio 30 | ratio 100 | ratio 300 |
|---|---|---|---|
| Asterix | 30.9 / 26.6 | 32.7 / 26.9 | 29.9 / 26.8 |
| Breakout | 15.9 / 10.6 | 14.7 / 11.3 | 13.5 / 11.4 |
| Freeway | 58.4 / 53.9 | 57.9 / 52.2 | 58.3 / 53.4 |
| Seaquest | 65.0 / 33.0 | 67.8 / 34.7 | 50.6 / 31.8 |
| SpaceInvaders | 140.4 / 100.3 | 129.4 / 100.7 | 127.5 / 110.4 |

## Conclusions

1. With equal tuning, QRC-ET (split form) is level with QRC on all five MinAtar games.
2. The Θ/Q learning-rate ratio makes little difference; a faster Θ (ratio 300) does not help, so Θ adapting too
   slowly to changing features is not the bottleneck.
3. Divergence: all 23 diverged runs (of 720) are QRC with λ = 0.9 or 0.95 (|Q| reached 1e4 to 1e18, one NaN, between
   270k and 480k steps; returns mostly collapsed). QRC-ET was only run with λ ≤ 0.8, and at λ = 0.24 and 0.8 neither
   method diverged, so this does NOT show that QRC-ET is more robust.
   Divergence definition: the logged Q(S_t, A_t) (scaled reward units, healthy runs ~1-3; one sample every 1000 steps)
   is ever non-finite or |Q| > 1e4. This is a fixed cut-off, not a derived bound.
4. QRC-ET's best λ was 0.24 (the shortest offered) on 4 of 5 games, while QRC often preferred λ=0.5, which QRC-ET
   did not get; if anything the grid favours QRC.
5. Together with R33-R35: expected traces clearly help with one-hot features (13-32% lower value error, every seed)
   and are neutral with conv features, consistent with the representation (feature collapse) explanation.
