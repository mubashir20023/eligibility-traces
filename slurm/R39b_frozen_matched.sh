#!/bin/bash
#SBATCH --job-name=R39b_matched
#SBATCH --array=0-799
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R39b_%A_%a.out
#SBATCH --error=logs/R39b_%A_%a.err
# R39b: Peter's suggestion 2 without confounds -- QRC vs QRC-ET on FROZEN pre-trained conv layers with IDENTICAL
# settings for both methods, so the trace is the only difference (R39 used each method's own tuned lr / lambda).
# Conv layers from the best R37 QRC run or the best R37 QRC-ET run (r39_sources.txt), fresh Q head, cleared traces.
#   q_lr {2.5e-4, 5e-4, 1e-3, 2e-3} x lambda {0.24, 0.8}; QRC-ET: split form, eta 0.3, Theta/Q ratio 100.
# 5 games x 2 sources x 8 settings x 2 methods x 5 seeds (501-505) = 800 runs.   Summary:  python summarize_r39b.py
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

GAMES=(Asterix Breakout Freeway Seaquest SpaceInvaders)
SRCS=(qrc et)
LRS=(2.5e-4 5e-4 1e-3 2e-3)
LAMS=(0.24 0.8)
METHODS=(qrc et)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 5 + 501));        T=$((T / 5))
METHOD=${METHODS[$((T % 2))]}; T=$((T / 2))
LR=${LRS[$((T % 4))]};        T=$((T / 4))
LAM=${LAMS[$((T % 2))]};      T=$((T / 2))
SRC=${SRCS[$((T % 2))]};      T=$((T / 2))
GAME=${GAMES[$T]}
MODEL=$(awk -v g=$GAME -v s=$SRC '$1==g && $2==s {print $3}' r39_sources.txt)
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --q_lr $LR --lamda $LAM --init_from $MODEL --freeze_body --resume"
NAME=R39b_${METHOD}_src${SRC}_lam${LAM}_lr${LR}
if [ "$METHOD" = "qrc" ]; then
  python algorithms/qrc.py $C --exp_name $NAME
else
  ETLR=$(python -c "print(100 * $LR)")
  python algorithms/qrc-et-algo3.py $C --et_lr $ETLR --et_algo3 --et_paired_rng --et_algo3_target raw \
    --no-et_algo3_lag --et_split --et_eta 0.3 --exp_name $NAME
fi
