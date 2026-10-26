#!/bin/bash
#SBATCH --job-name=R29_fw_lr
#SBATCH --array=0-89
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R29_fw_lr_%A_%a.out
#SBATCH --error=logs/R29_fw_lr_%A_%a.err
# R29: completes the learning-rate x agent grid on the SAME seeds as R28 (101-115).
# 6 arms x 15 seeds. Tasks 0-14 arm0, 15-29 arm1, 30-44 arm2, 45-59 arm3, 60-74 arm4, 75-89 arm5.

module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

SEEDS=(101 102 103 104 105 106 107 108 109 110 111 112 113 114 115)
ARM=$((SLURM_ARRAY_TASK_ID / 15))
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 15))]}
COMMON="--env_type minatar --env_id MinAtar/Freeway-v1 --seed $SEED --total_timesteps 5000000 --resume"
ET="--et_algo3 --et_eta 0.3 --et_paired_rng"

case $ARM in
  0) python algorithms/qrc.py $COMMON --exp_name R29_qrc_lam08_lr1e3 --q_lr 1e-3 --lamda 0.8 ;;
  1) python algorithms/qrc.py $COMMON --exp_name R29_qrc_lam08_lr2e3 --q_lr 2e-3 --lamda 0.8 ;;
  2) python algorithms/qrc.py $COMMON --exp_name R29_qrc_lam024_lr2e3 --q_lr 2e-3 --lamda 0.24 ;;
  3) python algorithms/qrc-et-algo3.py $COMMON --exp_name R29_et_eta03_lr5e4 --q_lr 5e-4 --lamda 0.8 $ET ;;
  4) python algorithms/qrc-et-algo3.py $COMMON --exp_name R29_et_eta03_lr1e3 --q_lr 1e-3 --lamda 0.8 $ET ;;
  5) python algorithms/qrc-et-algo3.py $COMMON --exp_name R29_et_eta03_lr4e3 --q_lr 4e-3 --lamda 0.8 $ET ;;
esac
