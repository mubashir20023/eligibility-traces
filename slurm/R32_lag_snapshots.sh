#!/bin/bash
#SBATCH --job-name=R32_lag
#SBATCH --array=0-74
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R32_lag_%A_%a.out
#SBATCH --error=logs/R32_lag_%A_%a.err
# R32: does applying the expected trace without Algorithm 3's one-step lag help, and what do the value maps look like?
# Multi-chain prediction, sigma = 1, m = 8, 32, 128 x 5 agents x seeds 1-5 = 75 tasks.
# Snapshots every 2000 steps and after episodes 1,3,10,30,100,300,1000,3000 (for the Fig. 3 style figures).
# Needs apply_eval_patch.py and apply_snapshot_patch.py applied.

module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

ENVS=(m8-n4-s1.0 m32-n4-s1.0 m128-n4-s1.0)
AGENTS=(qrc_lam0 qrc_lam08 et_lag et_nolag et_nolag_eta03)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 5 + 1));          T=$((T / 5))
AGENT=${AGENTS[$((T % 5))]};  T=$((T / 5))
ENV=${ENVS[$T]}
C="--env_type minatar --env_id multichain_env:MultiChainPred-${ENV}-v0 --seed $SEED --total_timesteps 100000 --eval_values --snapshot_every 2000 --snapshot_episodes 1,3,10,30,100,300,1000,3000 --resume --exp_name R32_${AGENT}"
ET="--et_algo3 --et_paired_rng --lamda 0.8 --q_lr 3e-3 --et_algo3_target raw"

case $AGENT in
  qrc_lam0)       python algorithms/qrc.py $C --lamda 0.0 --q_lr 3e-4 ;;
  qrc_lam08)      python algorithms/qrc.py $C --lamda 0.8 --q_lr 3e-3 ;;
  et_lag)         python algorithms/qrc-et-algo3.py $C $ET --et_eta 0.0 ;;
  et_nolag)       python algorithms/qrc-et-algo3.py $C $ET --et_eta 0.0 --no-et_algo3_lag ;;
  et_nolag_eta03) python algorithms/qrc-et-algo3.py $C $ET --et_eta 0.3 --no-et_algo3_lag ;;
esac
