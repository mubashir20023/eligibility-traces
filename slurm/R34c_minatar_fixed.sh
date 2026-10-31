#!/bin/bash
#SBATCH --job-name=R34c_minatar
#SBATCH --array=0-119
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R34c_%A_%a.out
#SBATCH --error=logs/R34c_%A_%a.err
# R34c: MinAtar with the two QRC-ET fixes (no one-step lag, ordinary-trace Theta target).
# Freeway, Asterix, Breakout x 4 agents x 2 learning rates x 5 fresh seeds (201-205) = 120 runs, 5M steps.
#   qrc_lam08, qrc_lam024        : QRC baselines (long and short trace)
#   et_fix_eta0, et_fix_eta03    : QRC-ET, no lag, raw target, eta 0 (pure expected trace) / 0.3 (mixture)
# Summary:  python summarize_r34c.py
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

GAMES=(Freeway Asterix Breakout)
AGENTS=(qrc_lam08 qrc_lam024 et_fix_eta0 et_fix_eta03)
LRS=(5e-4 2e-3)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 5 + 201));        T=$((T / 5))
LR=${LRS[$((T % 2))]};        T=$((T / 2))
AGENT=${AGENTS[$((T % 4))]};  T=$((T / 4))
GAME=${GAMES[$T]}

C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --q_lr $LR --resume --exp_name R34c_${AGENT}_lr${LR}"
ET="--et_algo3 --et_paired_rng --lamda 0.8 --et_algo3_target raw --no-et_algo3_lag"
case $AGENT in
  qrc_lam08)    python algorithms/qrc.py $C --lamda 0.8 ;;
  qrc_lam024)   python algorithms/qrc.py $C --lamda 0.24 ;;
  et_fix_eta0)  python algorithms/qrc-et-algo3.py $C $ET --et_eta 0.0 ;;
  et_fix_eta03) python algorithms/qrc-et-algo3.py $C $ET --et_eta 0.3 ;;
esac
