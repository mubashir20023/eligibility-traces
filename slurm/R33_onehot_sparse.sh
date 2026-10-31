#!/bin/bash
#SBATCH --job-name=R33_onehot
#SBATCH --array=0-719
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/R33_onehot_%A_%a.out
#SBATCH --error=logs/R33_onehot_%A_%a.err
# R33 (Peter + Mathieu): multi-chain PREDICTION with ONE-HOT input (no conv, no hidden layer, no bias, no obs-norm)
# and SPARSE terminal reward (+1 w.p. p, else 0, as in the ET paper's Fig. 1/2).
# 4 tasks (m = 8, 32 x p = 0.2, 0.9) x 6 agents x 3 learning rates x 10 seeds = 720 runs.
# Theta learning rate is set per task so each visit moves Theta a fixed fraction (beta) towards its target:
#   beta = et_lr * 2 / input_size  (input_size = 100 for m=8, 200 for m=32)
# Needs: multichain_env.py (with MultiChainSparse), apply_eval_patch.py, apply_snapshot_patch.py, apply_linear_patch.py.

module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

TASKS=(m8-n4-p0.2 m8-n4-p0.9 m32-n4-p0.2 m32-n4-p0.9)
AGENTS=(qrc_lam0 qrc_lam09 et_lag_default et_lag_b01 et_nolag_b01 et_nolag_b03)
LRS=(0.01 0.03 0.1)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 1));          T=$((T / 10))
LR=${LRS[$((T % 3))]};         T=$((T / 3))
AGENT=${AGENTS[$((T % 6))]};   T=$((T / 6))
TASK=${TASKS[$T]}
case $TASK in m8*) F=100 ;; m32*) F=200 ;; esac
B01=$(python -c "print(0.1 * $F / 2)"); B03=$(python -c "print(0.3 * $F / 2)")

C="--env_type minatar --env_id multichain_env:MultiChainSparse-${TASK}-v0 --seed $SEED --total_timesteps 120000 --net_arch linear --no-obs_norm --q_lr $LR --eval_values --snapshot_episodes 1,2,3,5,10,20,50,100,200,500,1000,2000,5000,10000,20000 --resume --exp_name R33_${AGENT}_lr${LR}"
ET="--et_head_name Dense_0 --et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 --et_algo3_target raw"

case $AGENT in
  qrc_lam0)       python algorithms/qrc.py $C --lamda 0.0 ;;
  qrc_lam09)      python algorithms/qrc.py $C --lamda 0.9 ;;
  et_lag_default) python algorithms/qrc-et-algo3.py $C $ET ;;
  et_lag_b01)     python algorithms/qrc-et-algo3.py $C $ET --et_lr $B01 ;;
  et_nolag_b01)   python algorithms/qrc-et-algo3.py $C $ET --et_lr $B01 --no-et_algo3_lag ;;
  et_nolag_b03)   python algorithms/qrc-et-algo3.py $C $ET --et_lr $B03 --no-et_algo3_lag ;;
esac
