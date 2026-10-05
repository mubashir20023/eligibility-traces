#!/bin/bash
#SBATCH --job-name=R13_checks
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R13_checks_%A_%a.out
#SBATCH --error=logs/R13_checks_%A_%a.err
#
# Check 1 (tasks 0-2): Algo-3 code with et_eta=1.0 (ET switched off) on Freeway.
#          Should reproduce plain QRC (Freeway ~53, seeds 50-58). If not, the
#          comparison against the QRC baselines is not like-for-like.
# Check 3 (tasks 3-5): Algo-3 with Theta learning rate 0.03 (R12 used 0.1).
# Submit:  sbatch slurm/R13_checks.sh

module load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs

SEEDS=(1 42 999)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
  EXP=R13_ctrl_eta1;  EXTRA="--et_eta 1.0"
else
  EXP=R13_lr003;      EXTRA="--et_eta 0.3 --et_lr 0.03"
fi

python algorithms/qrc-et-algo3.py \
  --env_type minatar --env_id MinAtar/Freeway-v1 --seed "$SEED" \
  --exp_name "$EXP" --total_timesteps 5000000 \
  --et_algo3 $EXTRA --resume
