#!/bin/bash
#SBATCH --job-name=R40_multihot
#SBATCH --array=0-1199
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/R40_%A_%a.out
#SBATCH --error=logs/R40_%A_%a.err
# R40: WHY does the expected-trace advantage disappear? Sparse multi-chain (m=32, n=4), linear agent on fixed codes:
# each state has a random binary code of length D with k ones (multichain_env.MultiChainCodeEnv).
# D=130,k=1 is one-hot (no sharing, the R33-R35 setting); D = 64, 32, 16 (k=3) and 10 (k=5) share more and more.
# Agents (as R34a / R35a): QRC lambda 0, QRC lambda 0.9, QRC-ET split (lambda 0.9, eta 0, Theta step ~0.1 per visit).
# 2 rewards (p 0.2, 0.9) x 5 codes x 3 agents x 4 learning rates x 10 seeds = 1200 runs, 120k steps (minutes each).
# Summary:  python summarize_r40.py
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

PS=(0.2 0.9)
CODES=("130 1" "64 3" "32 3" "16 3" "10 5")
AGENTS=(qrc_lam0 qrc_lam09 et_split)
LRS=(0.003 0.01 0.03 0.1)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 1));          T=$((T / 10))
LR=${LRS[$((T % 4))]};         T=$((T / 4))
AGENT=${AGENTS[$((T % 3))]};   T=$((T / 3))
read D K <<< "${CODES[$((T % 5))]}"; T=$((T / 5))
P=${PS[$T]}
F=$(( ( (D + 99) / 100 ) * 100 ))                 # flattened input size the agent sees
ETLR=$(python -c "print(0.1 * $F / (2 * $K))")    # Theta step ~0.1 per visit for k ones (R33: 0.1*F/2 for one-hot)

C="--env_type minatar --env_id multichain_env:MultiChainCode-m32-n4-p${P}-D${D}-k${K}-v0 --seed $SEED --total_timesteps 120000 --net_arch linear --no-obs_norm --q_lr $LR --eval_values --resume --exp_name R40_${AGENT}_lr${LR}"
case $AGENT in
  qrc_lam0)  python algorithms/qrc.py $C --lamda 0.0 ;;
  qrc_lam09) python algorithms/qrc.py $C --lamda 0.9 ;;
  et_split)  python algorithms/qrc-et-algo3.py $C --et_head_name Dense_0 --et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 \
               --et_algo3_target raw --no-et_algo3_lag --et_split --et_lr $ETLR ;;
esac
