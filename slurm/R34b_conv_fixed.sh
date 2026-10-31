#!/bin/bash
#SBATCH --job-name=R34b_conv
#SBATCH --array=0-599
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R34b_%A_%a.out
#SBATCH --error=logs/R34b_%A_%a.err
# R34b: do the two fixes (no lag + Theta learning rate) also help with the normal MinAtar CONV network?
# Same sparse multi-chain tasks as R33 (m = 8, 32 x p = 0.2, 0.9), conv network and observation normalisation as usual.
# 5 agents x 3 learning rates x 10 seeds x 4 tasks = 600 runs.   Summary:  python summarize_mc.py R34b
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

TASKS=(m8-n4-p0.2 m8-n4-p0.9 m32-n4-p0.2 m32-n4-p0.9)
AGENTS=(qrc_lam0 qrc_lam09 et_lag_default et_nolag_etlr01 et_nolag_etlr03)
LRS=(3e-4 1e-3 3e-3)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 1));          T=$((T / 10))
LR=${LRS[$((T % 3))]};         T=$((T / 3))
AGENT=${AGENTS[$((T % 5))]};   T=$((T / 5))
TASK=${TASKS[$T]}

C="--env_type minatar --env_id multichain_env:MultiChainSparse-${TASK}-v0 --seed $SEED --total_timesteps 120000 --q_lr $LR --eval_values --snapshot_episodes 1,10,100,1000,10000 --resume --exp_name R34b_${AGENT}_lr${LR}"
ET="--et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 --et_algo3_target raw"
case $AGENT in
  qrc_lam0)        python algorithms/qrc.py $C --lamda 0.0 ;;
  qrc_lam09)       python algorithms/qrc.py $C --lamda 0.9 ;;
  et_lag_default)  python algorithms/qrc-et-algo3.py $C $ET ;;
  et_nolag_etlr01) python algorithms/qrc-et-algo3.py $C $ET --et_lr 0.1 --no-et_algo3_lag ;;
  et_nolag_etlr03) python algorithms/qrc-et-algo3.py $C $ET --et_lr 0.3 --no-et_algo3_lag ;;
esac
