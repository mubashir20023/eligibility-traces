#!/bin/bash
#SBATCH --job-name=R15_zero_theta
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R15_zero_theta_%A_%a.out
#SBATCH --error=logs/R15_zero_theta_%A_%a.err
#
# ROOT-CAUSE ABLATION (Freeway): Theta's prediction is replaced by ZERO in the Eq.3 mixture
# (--et_zero_theta), so y_t = eta * e_w_t. Same shortened head-trace decay as real ET
# (eta*gamma*lambda per step), but Theta has no influence.
#   tasks 0-4: eta 0.3 (compare with R12_et_algo3, Freeway, n=5: 39.37)
#   tasks 5-9: eta 0.6 (compare with R14_eta06, n=3: 41.90)
# Reading: zero-Theta ~ real ET  -> Theta contributes nothing; the loss is the mixing shortening the trace
#          zero-Theta WORSE      -> Theta is helping, but not enough
#          zero-Theta BETTER     -> Theta's prediction is actively harmful

module load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs

SEEDS=(1 2 3 42 999)
IDX=$((SLURM_ARRAY_TASK_ID % 5))
SEED=${SEEDS[$IDX]}
if [ "$SLURM_ARRAY_TASK_ID" -lt 5 ]; then EXP=R15_zero_eta03; ETA=0.3; else EXP=R15_zero_eta06; ETA=0.6; fi

python algorithms/qrc-et-algo3.py \
  --env_type minatar --env_id MinAtar/Freeway-v1 --seed "$SEED" \
  --exp_name "$EXP" --total_timesteps 5000000 \
  --et_algo3 --et_zero_theta --et_eta "$ETA" --resume
