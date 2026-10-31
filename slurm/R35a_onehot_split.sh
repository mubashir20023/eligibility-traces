#!/bin/bash
#SBATCH --job-name=R35a_split
#SBATCH --array=0-79
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/R35a_%A_%a.out
#SBATCH --error=logs/R35a_%A_%a.err
# R35a: one-hot sanity check of --et_split. Same tasks, seeds (11-30), learning rates and Theta rate as the R34a winner
# (et_nolag_b01), so the split form should give the same result.  4 tasks x 20 seeds = 80 runs.
# Summary (pooled with the R34a QRC runs):  python summarize_mc.py R34a R35a
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

CONFIGS=("m8-n4-p0.2 0.01" "m8-n4-p0.9 0.03" "m32-n4-p0.2 0.03" "m32-n4-p0.9 0.1")
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 20 + 11))
read TASK LR <<< "${CONFIGS[$((T / 20))]}"
case $TASK in m8*) F=100 ;; m32*) F=200 ;; esac
B01=$(python -c "print(0.1 * $F / 2)")

python algorithms/qrc-et-algo3.py --env_type minatar --env_id multichain_env:MultiChainSparse-${TASK}-v0 --seed $SEED \
  --total_timesteps 120000 --net_arch linear --no-obs_norm --q_lr $LR --eval_values --resume --exp_name R35a_et_split_b01_lr${LR} \
  --et_head_name Dense_0 --et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 --et_algo3_target raw --et_lr $B01 \
  --no-et_algo3_lag --et_split
