#!/bin/bash
#SBATCH --job-name=R30_pred
#SBATCH --array=0-599
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R30_pred_%A_%a.out
#SBATCH --error=logs/R30_pred_%A_%a.err
# R30: multi-chain PREDICTION screen (the paper's task) with the production agents + --eval_values.
# 8 envs (m = 1, 8, 32, 128 chains x reward noise sigma = 0, 1) x 5 agents x 3 learning rates x 5 seeds = 600.
# Metric: value RMSE over ALL states vs exact values (lower = better). m = 1 is the control (no merging paths).
# Pass condition: ET below the best-tuned QRC (any lambda) at m >= 32, as tabular ET is.
# Needs multichain_env.py in the repo root and apply_eval_patch.py already applied.

module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

ENVS=(m1-n4-s0.0 m8-n4-s0.0 m32-n4-s0.0 m128-n4-s0.0 m1-n4-s1.0 m8-n4-s1.0 m32-n4-s1.0 m128-n4-s1.0)
AGENTS=(qrc_lam0 qrc_lam024 qrc_lam08 et_eta0 et_eta03)
LRS=(3e-4 1e-3 3e-3)
SEEDS=(1 2 3 4 5)

T=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((T % 5))]};     T=$((T / 5))
LR=${LRS[$((T % 3))]};         T=$((T / 3))
AGENT=${AGENTS[$((T % 5))]};   T=$((T / 5))
ENV=${ENVS[$T]}

COMMON="--env_type minatar --env_id multichain_env:MultiChainPred-${ENV}-v0 --seed $SEED --total_timesteps 100000 --q_lr $LR --eval_values --resume --exp_name R30_${AGENT}_lr${LR}"
ET="--et_algo3 --et_paired_rng --lamda 0.8"

case $AGENT in
  qrc_lam0)   python algorithms/qrc.py $COMMON --lamda 0.0 ;;
  qrc_lam024) python algorithms/qrc.py $COMMON --lamda 0.24 ;;
  qrc_lam08)  python algorithms/qrc.py $COMMON --lamda 0.8 ;;
  et_eta0)    python algorithms/qrc-et-algo3.py $COMMON $ET --et_eta 0.0 ;;
  et_eta03)   python algorithms/qrc-et-algo3.py $COMMON $ET --et_eta 0.3 ;;
esac
