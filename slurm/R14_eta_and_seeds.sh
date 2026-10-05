#!/bin/bash
#SBATCH --job-name=R14_eta_seeds
#SBATCH --array=0-9
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R14_eta_seeds_%A_%a.out
#SBATCH --error=logs/R14_eta_seeds_%A_%a.err
#
# Freeway only.
#  tasks 0-2 : Algo 3, et_eta 0.6  (less Theta)      seeds 1 42 999   exp R14_eta06
#  tasks 3-5 : Algo 3, et_eta 0.8  (even less Theta) seeds 1 42 999   exp R14_eta08
#  tasks 6-7 : control (et_eta 1.0)  extra seeds 2 3  -> joins exp R13_ctrl_eta1 (n=5)
#  tasks 8-9 : R12 setting (et_eta 0.3) extra seeds 2 3 -> joins exp R12_et_algo3 (n=5 on Freeway)
# Question: is Freeway performance monotone in how much Theta is trusted
# (eta 0.3 < 0.6 < 0.8 < 1.0)?  And is the 0.3-vs-1.0 gap real at n=5?

module load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs

case $SLURM_ARRAY_TASK_ID in
  0|1|2) EXP=R14_eta06;      EXTRA="--et_eta 0.6"; SEEDS=(1 42 999); IDX=$SLURM_ARRAY_TASK_ID ;;
  3|4|5) EXP=R14_eta08;      EXTRA="--et_eta 0.8"; SEEDS=(1 42 999); IDX=$((SLURM_ARRAY_TASK_ID-3)) ;;
  6|7)   EXP=R13_ctrl_eta1;  EXTRA="--et_eta 1.0"; SEEDS=(2 3);      IDX=$((SLURM_ARRAY_TASK_ID-6)) ;;
  8|9)   EXP=R12_et_algo3;   EXTRA="--et_eta 0.3"; SEEDS=(2 3);      IDX=$((SLURM_ARRAY_TASK_ID-8)) ;;
esac
SEED=${SEEDS[$IDX]}

python algorithms/qrc-et-algo3.py \
  --env_type minatar --env_id MinAtar/Freeway-v1 --seed "$SEED" \
  --exp_name "$EXP" --total_timesteps 5000000 \
  --et_algo3 $EXTRA --resume
