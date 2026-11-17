#!/bin/bash
#SBATCH --job-name=R40b_tuning
#SBATCH --array=0-1119
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/R40b_%A_%a.out
#SBATCH --error=logs/R40b_%A_%a.err
# R40b: fairness check for R40 on the SHARED codes (D = 64, 32, 16, 10). In R40 the best QRC-ET learning rate was the
# lowest one tried (0.003) and Theta's step was fixed. Here: two lower learning rates for every agent, and Theta's step
# x0.3 / x3 for QRC-ET. Pooled with R40 by:  python summarize_r40.py R40 R40b
#   QRC lambda 0, QRC lambda 0.9          : lr {0.0003, 0.001}                                 (4 settings)
#   QRC-ET split, Theta step x1           : lr {0.0003, 0.001}                                 (2 settings)
#   QRC-ET split, Theta step x0.3 and x3  : lr {0.0003, 0.001, 0.003, 0.01}                    (8 settings)
# 2 rewards x 4 codes x 14 settings x 10 seeds = 1120 runs, 120k steps (minutes each).
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

PS=(0.2 0.9)
CODES=("64 3" "32 3" "16 3" "10 5")
SETTINGS=("qrc_lam0 0.0003 1" "qrc_lam0 0.001 1" "qrc_lam09 0.0003 1" "qrc_lam09 0.001 1"
          "et_split 0.0003 1" "et_split 0.001 1"
          "et_split_th03 0.0003 0.3" "et_split_th03 0.001 0.3" "et_split_th03 0.003 0.3" "et_split_th03 0.01 0.3"
          "et_split_th3 0.0003 3" "et_split_th3 0.001 3" "et_split_th3 0.003 3" "et_split_th3 0.01 3")
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 1));                      T=$((T / 10))
read AGENT LR MULT <<< "${SETTINGS[$((T % 14))]}"; T=$((T / 14))
read D K <<< "${CODES[$((T % 4))]}";       T=$((T / 4))
P=${PS[$T]}
F=$(( ( (D + 99) / 100 ) * 100 ))
ETLR=$(python -c "print($MULT * 0.1 * $F / (2 * $K))")

C="--env_type minatar --env_id multichain_env:MultiChainCode-m32-n4-p${P}-D${D}-k${K}-v0 --seed $SEED --total_timesteps 120000 --net_arch linear --no-obs_norm --q_lr $LR --eval_values --resume --exp_name R40b_${AGENT}_lr${LR}"
case $AGENT in
  qrc_lam0)  python algorithms/qrc.py $C --lamda 0.0 ;;
  qrc_lam09) python algorithms/qrc.py $C --lamda 0.9 ;;
  et_*)      python algorithms/qrc-et-algo3.py $C --et_head_name Dense_0 --et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 \
               --et_algo3_target raw --no-et_algo3_lag --et_split --et_lr $ETLR ;;
esac
