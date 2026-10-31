#!/bin/bash
#SBATCH --job-name=R35_minatar
#SBATCH --array=0-89
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R35_%A_%a.out
#SBATCH --error=logs/R35_%A_%a.err
# R35: MinAtar with the stable split form of QRC-ET (no lag, exact current gradient + expected PAST trace).
# Same games, learning rates and seeds (201-205) as R34c, so the R34c QRC runs are the baseline (paired seeds).
#   et_split_eta0, et_split_eta03 : --et_split --no-et_algo3_lag, eta 0 / 0.3
#   et_lag_eta03                  : current code (Algorithm 3 with the lag), the stable version used in R29
# 3 games x 3 agents x 2 learning rates x 5 seeds = 90 runs, 5M steps.
# Summary (pooled with R34c):  python summarize_r34c.py R34c R35
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

GAMES=(Freeway Asterix Breakout)
AGENTS=(et_split_eta0 et_split_eta03 et_lag_eta03)
LRS=(5e-4 2e-3)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 5 + 201));        T=$((T / 5))
LR=${LRS[$((T % 2))]};        T=$((T / 2))
AGENT=${AGENTS[$((T % 3))]};  T=$((T / 3))
GAME=${GAMES[$T]}

C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --q_lr $LR --resume --exp_name R35_${AGENT}_lr${LR}"
ET="--et_algo3 --et_paired_rng --lamda 0.8 --et_algo3_target raw"
case $AGENT in
  et_split_eta0)  python algorithms/qrc-et-algo3.py $C $ET --no-et_algo3_lag --et_split --et_eta 0.0 ;;
  et_split_eta03) python algorithms/qrc-et-algo3.py $C $ET --no-et_algo3_lag --et_split --et_eta 0.3 ;;
  et_lag_eta03)   python algorithms/qrc-et-algo3.py $C $ET --et_eta 0.3 ;;
esac
