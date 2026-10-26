#!/bin/bash
#SBATCH --job-name=R27_fw_2e3
#SBATCH --array=0-29
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R27_fw_2e3_%A_%a.out
#SBATCH --error=logs/R27_fw_2e3_%A_%a.err
#
# R26 (Freeway, 5 seeds): Algo 3 peaks at q_lr 2e-3 (59.2) and QRC is flat at 52-54 (54.2 at 2e-3); 15-seed best-vs-best at 1e-3 / 5e-4 was a tie.
# The lr was picked on seeds 1 2 3 42 999, so test it on FRESH seeds 4..13 (no selection bias):
#   tasks  0- 9: Freeway plain QRC             q_lr 2e-3   seeds 4..13   exp R27_qrc_lr2e3_more
#   tasks 10-19: Freeway Algo 3 (eta 0.3, paired) q_lr 2e-3 seeds 4..13   exp R27_algo3_lr2e3_more
# And the same lr on Asterix, seeds 1 2 3 42 999 (does Algo 3 rise there too?):
#   tasks 20-24: Asterix plain QRC             q_lr 2e-3   exp R27_qrc_lr2e3
#   tasks 25-29: Asterix Algo 3 (eta 0.3)      q_lr 2e-3   exp R27_algo3_lr2e3
# Read with: python summarize_27.py

module load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs

S5=(1 2 3 42 999)
S10=(4 5 6 7 8 9 10 11 12 13)
T=$SLURM_ARRAY_TASK_ID
if   [ "$T" -lt 10 ]; then ENV=Freeway; SEED=${S10[$T]};        ALGO=qrc;   EXP=R27_qrc_lr2e3_more
elif [ "$T" -lt 20 ]; then ENV=Freeway; SEED=${S10[$((T-10))]}; ALGO=algo3; EXP=R27_algo3_lr2e3_more
elif [ "$T" -lt 25 ]; then ENV=Asterix; SEED=${S5[$((T-20))]};  ALGO=qrc;   EXP=R27_qrc_lr2e3
else                       ENV=Asterix; SEED=${S5[$((T-25))]};  ALGO=algo3; EXP=R27_algo3_lr2e3
fi
COMMON="--env_type minatar --env_id MinAtar/${ENV}-v1 --seed $SEED --total_timesteps 5000000 --q_lr 2e-3 --resume --exp_name $EXP"

if [ "$ALGO" = "qrc" ]; then
  python algorithms/qrc.py $COMMON
else
  python algorithms/qrc-et-algo3.py $COMMON --et_algo3 --et_eta 0.3 --et_paired_rng
fi
