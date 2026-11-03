#!/bin/bash
#SBATCH --job-name=R36_tune
#SBATCH --array=0-719
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R36_%A_%a.out
#SBATCH --error=logs/R36_%A_%a.err
# R36 (stage 1 of 2): equal-budget hyperparameter search on all five MinAtar games (Peter's suggestion).
# Both methods get the SAME learning-rate grid and the SAME number of configurations (24 each):
#   QRC     : q_lr {2.5e-4, 5e-4, 1e-3, 2e-3} x lambda {0, 0.24, 0.5, 0.8, 0.9, 0.95}
#   QRC-ET  : q_lr {2.5e-4, 5e-4, 1e-3, 2e-3} x lambda {0.24, 0.8} x Theta/Q rate ratio {30, 100, 300}
#             (split form, no lag, eta 0.3; Theta learning rate = ratio * q_lr, so the old default 0.1 is ratio 100 at 1e-3)
# 5 games x 48 configs x 3 seeds (301-303) = 720 runs, 5M steps. Stage 2 (R37) re-runs each method's best
# configuration per game on 10 fresh seeds.   Summary + stage-2 list:  python summarize_r36.py
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

GAMES=(Freeway Breakout Asterix Seaquest SpaceInvaders)
LRS=(2.5e-4 5e-4 1e-3 2e-3)
QLAMS=(0 0.24 0.5 0.8 0.9 0.95)
ELAMS=(0.24 0.8)
RATIOS=(30 100 300)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 3 + 301));  T=$((T / 3))
CFG=$((T % 48));        T=$((T / 48))
GAME=${GAMES[$T]}
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --resume"
if [ $CFG -lt 24 ]; then                                   # QRC: 4 lr x 6 lambda
  LR=${LRS[$((CFG % 4))]}; LAM=${QLAMS[$((CFG / 4))]}
  python algorithms/qrc.py $C --q_lr $LR --lamda $LAM --exp_name R36_qrc_lam${LAM}_lr${LR}
else                                                       # QRC-ET: 4 lr x 2 lambda x 3 ratio
  K=$((CFG - 24)); LR=${LRS[$((K % 4))]}; LAM=${ELAMS[$((K / 4 % 2))]}; R=${RATIOS[$((K / 8))]}
  ETLR=$(python -c "print($R * $LR)")
  python algorithms/qrc-et-algo3.py $C --q_lr $LR --lamda $LAM --et_lr $ETLR \
    --et_algo3 --et_paired_rng --et_algo3_target raw --no-et_algo3_lag --et_split --et_eta 0.3 \
    --exp_name R36_et_lam${LAM}_r${R}_lr${LR}
fi
