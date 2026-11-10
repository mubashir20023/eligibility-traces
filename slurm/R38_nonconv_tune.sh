#!/bin/bash
#SBATCH --job-name=R38_tune
#SBATCH --array=0-719
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R38_%A_%a.out
#SBATCH --error=logs/R38_%A_%a.err
# R38 (stage 1): Peter's suggestion 1 -- MinAtar WITHOUT convolution, equal-budget tuning exactly as R36.
# Submit once per architecture:
#   sbatch --export=ALL,ARCH=linear R38_nonconv_tune.sh    (Q = W x on the flattened image)
#   sbatch --export=ALL,ARCH=mlp    R38_nonconv_tune.sh    (flatten -> Dense 128 -> Q head)
# Both methods: same 4 learning rates, 24 settings each (QRC: 6 lambda; QRC-ET split, eta 0.3: 2 lambda x 3 Theta/Q ratios).
# 5 games x 48 settings x 3 seeds (301-303) = 720 runs per architecture.
# Summary:  python summarize_r36.py R38lin   |   python summarize_r36.py R38mlp   (writes r38lin_best.txt / r38mlp_best.txt)
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

case $ARCH in
  linear) PRE=R38lin2; LRS=(2.5e-3 5e-3 1e-2 2e-2); HEAD="--et_head_name Dense_0" ;;  # raw 0/1 pixels   # linear: ~4x larger effective step, grid shifted down
  mlp)    PRE=R38mlp; LRS=(2.5e-4 5e-4 1e-3 2e-3);    HEAD="" ;;
  *) echo "set ARCH=linear or ARCH=mlp"; exit 1 ;;
esac
[ "$ARCH" = "linear" ] && NORM="--no-obs_norm" || NORM=""
GAMES=(Freeway Breakout Asterix Seaquest SpaceInvaders)
QLAMS=(0 0.24 0.5 0.8 0.9 0.95)
ELAMS=(0.24 0.8)
RATIOS=(30 100 300)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 3 + 301));  T=$((T / 3))
CFG=$((T % 48));        T=$((T / 48))
GAME=${GAMES[$T]}
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --net_arch $ARCH $NORM --resume"
if [ $CFG -lt 24 ]; then
  LR=${LRS[$((CFG % 4))]}; LAM=${QLAMS[$((CFG / 4))]}
  python algorithms/qrc.py $C --q_lr $LR --lamda $LAM --exp_name ${PRE}_qrc_lam${LAM}_lr${LR}
else
  K=$((CFG - 24)); LR=${LRS[$((K % 4))]}; LAM=${ELAMS[$((K / 4 % 2))]}; R=${RATIOS[$((K / 8))]}
  ETLR=$(python -c "print($R * $LR)")
  python algorithms/qrc-et-algo3.py $C $HEAD --q_lr $LR --lamda $LAM --et_lr $ETLR \
    --et_algo3 --et_paired_rng --et_algo3_target raw --no-et_algo3_lag --et_split --et_eta 0.3 \
    --exp_name ${PRE}_et_lam${LAM}_r${R}_lr${LR}
fi
