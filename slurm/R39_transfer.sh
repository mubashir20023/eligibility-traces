#!/bin/bash
#SBATCH --job-name=R39_transfer
#SBATCH --array=0-399
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R39_%A_%a.out
#SBATCH --error=logs/R39_%A_%a.err
# R39: Peter's suggestion 2 -- QRC vs QRC-ET on a PRE-TRAINED representation.
# Conv layers copied from the best R37 run (r39_sources.txt; source = best QRC run or best QRC-ET run, so neither method
# only gets its own features), Q head freshly initialised, traces / Theta / optimiser state cleared.
# Conv layers frozen (frz) or still learning (unf). Each method uses its R37 setting for that game (r36_best.txt).
# 5 games x 2 sources x 2 (frozen / unfrozen) x 2 methods x 10 seeds (501-510) = 400 runs, 5M steps.
# Run first:  python prepare_r39.py        Summary:  python summarize_r39.py
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

GAMES=(Asterix Breakout Freeway Seaquest SpaceInvaders)
SRCS=(qrc et)
MODES=(frz unf)
METHODS=(qrc et)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 501));      T=$((T / 10))
METHOD=${METHODS[$((T % 2))]}; T=$((T / 2))
MODE=${MODES[$((T % 2))]};     T=$((T / 2))
SRC=${SRCS[$((T % 2))]};       T=$((T / 2))
GAME=${GAMES[$T]}
MODEL=$(awk -v g=$GAME -v s=$SRC '$1==g && $2==s {print $3}' r39_sources.txt)
read G M LAM R LR <<< "$(awk -v g=$GAME -v m=$METHOD '$1==g && $2==m' r36_best.txt)"
FRZ=""; [ "$MODE" = "frz" ] && FRZ="--freeze_body"
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --q_lr $LR --lamda $LAM --init_from $MODEL $FRZ --resume"
NAME=R39_${METHOD}_src${SRC}_${MODE}_lr${LR}
if [ "$METHOD" = "qrc" ]; then
  python algorithms/qrc.py $C --exp_name $NAME
else
  ETLR=$(python -c "print($R * $LR)")
  python algorithms/qrc-et-algo3.py $C --et_lr $ETLR --et_algo3 --et_paired_rng --et_algo3_target raw \
    --no-et_algo3_lag --et_split --et_eta 0.3 --exp_name $NAME
fi
