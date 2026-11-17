#!/bin/bash
#SBATCH --job-name=R43_lam
#SBATCH --array=0-899
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R43_%A_%a.out
#SBATCH --error=logs/R43_%A_%a.err
# R43: the missing MinAtar cells. In R36 QRC-ET ran only at lambda 0.24 and 0.8, with no control, and only in the split form.
# The expected-traces paper compares at ONE long trace (lambda 0.95), where QRC collapses in R36.
# Same games, learning rates and seeds (301-303) as R36, so the R36 QRC runs are the baseline (paired seeds).
# All arms: theta learning rate = 100 x Q learning rate (as R36 ratio 100); eta 0.3 unless stated.
#   et     : QRC-ET split form (the R36 agent)           at lambda 0.5, 0.9, 0.95    (0.24 and 0.8 are in R36)
#   zero   : the same agent with theta's output set to 0 at lambda 0.24, 0.5, 0.8, 0.9, 0.95
#            (control: the head trace is only shortened to eta x lambda; nothing is learned by theta)
#   alg3   : the paper's Algorithm 3 form (one-step lag, theta trained on the mixed trace) at the same five lambda
#   alg3e0 : Algorithm 3 with eta 0 (the pure expected trace, the paper's Fig. 6 setting) at lambda 0.95 and 0.8
# 15 arms x 5 games x 4 learning rates x 3 seeds = 900 runs, 5M steps. Order: et, zero, alg3, alg3e0; long lambda first.
# Summary (pooled with R36):  python summarize_lambda.py R36 R43
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

GAMES=(Freeway Breakout Asterix Seaquest SpaceInvaders)
LRS=(2.5e-4 5e-4 1e-3 2e-3)
ARMS=("et 0.95" "et 0.9" "et 0.5"
      "zero 0.95" "zero 0.9" "zero 0.8" "zero 0.5" "zero 0.24"
      "alg3 0.95" "alg3 0.9" "alg3 0.8" "alg3 0.5" "alg3 0.24"
      "alg3e0 0.95" "alg3e0 0.8")
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 3 + 301));       T=$((T / 3))
LR=${LRS[$((T % 4))]};       T=$((T / 4))
GAME=${GAMES[$((T % 5))]};   T=$((T / 5))
read ARM LAM <<< "${ARMS[$T]}"
ETLR=$(python -c "print(100 * $LR)")

C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --resume --q_lr $LR --lamda $LAM --et_lr $ETLR --et_algo3 --et_paired_rng --exp_name R43_${ARM}_lam${LAM}_r100_lr${LR}"
case $ARM in
  et)     python algorithms/qrc-et-algo3.py $C --et_eta 0.3 --et_algo3_target raw --no-et_algo3_lag --et_split ;;
  zero)   python algorithms/qrc-et-algo3.py $C --et_eta 0.3 --et_algo3_target raw --no-et_algo3_lag --et_split --et_zero_theta ;;
  alg3)   python algorithms/qrc-et-algo3.py $C --et_eta 0.3 --et_algo3_target e_w ;;
  alg3e0) python algorithms/qrc-et-algo3.py $C --et_eta 0.0 --et_algo3_target e_w ;;
esac
