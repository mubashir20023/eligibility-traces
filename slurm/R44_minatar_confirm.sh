#!/bin/bash
#SBATCH --job-name=R44_conf
#SBATCH --array=0-299
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R44_%A_%a.out
#SBATCH --error=logs/R44_%A_%a.err
# R44: 10-seed confirmation of the R43 MinAtar screen (3 seeds). Settings from r44_best.txt (python prepare_r44.py):
# per game, QRC and QRC-ET at lambda 0.9, at lambda 0.95, and each method's best setting overall.
# 5 games x 6 configurations x 10 FRESH seeds (601-610) = 300 runs, 5M steps.   Summary:  python summarize_r44.py
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 601))
read GAME METHOD LAM R LR TAG <<< "$(sed -n "$((T / 10 + 1))p" r44_best.txt)"
if [ -z "$GAME" ]; then echo "no line $((T / 10 + 1)) in r44_best.txt"; exit 1; fi
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --q_lr $LR --lamda $LAM --resume"
if [ "$METHOD" = "qrc" ]; then
  python algorithms/qrc.py $C --exp_name R44_${TAG}_qrc_lam${LAM}_lr${LR}
else
  ETLR=$(python -c "print($R * $LR)")
  python algorithms/qrc-et-algo3.py $C --et_lr $ETLR --et_algo3 --et_paired_rng --et_algo3_target raw \
    --no-et_algo3_lag --et_split --et_eta 0.3 --exp_name R44_${TAG}_et_lam${LAM}_r${R}_lr${LR}
fi
