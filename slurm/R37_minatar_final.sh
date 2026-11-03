#!/bin/bash
#SBATCH --job-name=R37_final
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R37_%A_%a.out
#SBATCH --error=logs/R37_%A_%a.err
# R37 (stage 2 of 2): best QRC and best QRC-ET configuration per game from R36 (r36_best.txt), 10 FRESH seeds (401-410).
# 5 games x 2 methods x 10 seeds = 100 runs. Run after:  python summarize_r36.py
# Summary:  python summarize_r34c.py R37
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 401))
read GAME METHOD LAM R LR <<< "$(sed -n "$((T / 10 + 1))p" r36_best.txt)"
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --q_lr $LR --lamda $LAM --resume"
if [ "$METHOD" = "qrc" ]; then
  python algorithms/qrc.py $C --exp_name R37_qrc_best_lr${LR}
else
  ETLR=$(python -c "print($R * $LR)")
  python algorithms/qrc-et-algo3.py $C --et_lr $ETLR --et_algo3 --et_paired_rng --et_algo3_target raw \
    --no-et_algo3_lag --et_split --et_eta 0.3 --exp_name R37_et_best_lr${LR}
fi
