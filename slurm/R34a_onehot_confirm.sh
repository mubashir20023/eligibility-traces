#!/bin/bash
#SBATCH --job-name=R34a_confirm
#SBATCH --array=0-419
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/R34a_%A_%a.out
#SBATCH --error=logs/R34a_%A_%a.err
# R34a: FRESH-SEED confirmation of R33 (one-hot input, sparse reward). Seeds 11-30 (never used before).
# Each agent at the learning rate R33 selected; where R33's best was the top of the grid (0.1), 0.3 is added
# so that the best QRC is not limited by the grid. 21 configurations x 20 seeds = 420 runs.
# Summary:  python summarize_mc.py R34a
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

CONFIGS=(
  "m8-n4-p0.2 qrc_lam0 0.03"   "m8-n4-p0.2 qrc_lam09 0.01"  "m8-n4-p0.2 et_nolag_b01 0.01"  "m8-n4-p0.2 et_lag_default 0.01"
  "m8-n4-p0.9 qrc_lam0 0.1"    "m8-n4-p0.9 qrc_lam0 0.3"    "m8-n4-p0.9 qrc_lam09 0.03"     "m8-n4-p0.9 et_nolag_b01 0.03"  "m8-n4-p0.9 et_lag_default 0.03"
  "m32-n4-p0.2 qrc_lam0 0.1"   "m32-n4-p0.2 qrc_lam0 0.3"   "m32-n4-p0.2 qrc_lam09 0.03"    "m32-n4-p0.2 et_nolag_b01 0.03" "m32-n4-p0.2 et_lag_default 0.03"
  "m32-n4-p0.9 qrc_lam0 0.1"   "m32-n4-p0.9 qrc_lam0 0.3"   "m32-n4-p0.9 qrc_lam09 0.1"     "m32-n4-p0.9 qrc_lam09 0.3"
  "m32-n4-p0.9 et_nolag_b01 0.1" "m32-n4-p0.9 et_nolag_b01 0.3" "m32-n4-p0.9 et_lag_default 0.1"
)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 20 + 11))
read TASK AGENT LR <<< "${CONFIGS[$((T / 20))]}"
case $TASK in m8*) F=100 ;; m32*) F=200 ;; esac
B01=$(python -c "print(0.1 * $F / 2)")

C="--env_type minatar --env_id multichain_env:MultiChainSparse-${TASK}-v0 --seed $SEED --total_timesteps 120000 --net_arch linear --no-obs_norm --q_lr $LR --eval_values --resume --exp_name R34a_${AGENT}_lr${LR}"
ET="--et_head_name Dense_0 --et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 --et_algo3_target raw"
case $AGENT in
  qrc_lam0)       python algorithms/qrc.py $C --lamda 0.0 ;;
  qrc_lam09)      python algorithms/qrc.py $C --lamda 0.9 ;;
  et_lag_default) python algorithms/qrc-et-algo3.py $C $ET ;;
  et_nolag_b01)   python algorithms/qrc-et-algo3.py $C $ET --et_lr $B01 --no-et_algo3_lag ;;
esac
