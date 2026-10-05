#!/bin/bash
#SBATCH --job-name=R12_et_algo3
#SBATCH --array=0-14
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R12_et_algo3_%A_%a.out
#SBATCH --error=logs/R12_et_algo3_%A_%a.err
# ^ copy partition/account/--gres lines from your working slurm/R9_et_allcol.sh
#
# 5 MinAtar games x 3 seeds = 15 independent array tasks (NOT srun-in-a-loop,
# which ran serially per your srun_par_test).
# Submit:  mkdir -p logs && sbatch slurm/R12_et_algo3.sh

module load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs

ENVS=("MinAtar/Freeway-v1" "MinAtar/Seaquest-v1" "MinAtar/Asterix-v1" "MinAtar/Breakout-v1" "MinAtar/SpaceInvaders-v1")
SEEDS=(1 42 999)
ENV=${ENVS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

# Unique --exp_name => unique checkpoint dir => cannot silently resume an old run.
# CSV (training_data.csv) has all metrics, so wandb tracking is not needed.
python algorithms/qrc-et-algo3.py \
  --env_type minatar --env_id "$ENV" --seed "$SEED" \
  --exp_name R12_et_algo3 \
  --total_timesteps 5000000 \
  --et_algo3 --et_eta 0.3 \
  --resume
