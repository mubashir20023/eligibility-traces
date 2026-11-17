#!/bin/bash
#SBATCH --job-name=R40c_oracle
#SBATCH --array=0-599
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --mem=8G
#SBATCH --output=logs/R40c_%A_%a.out
#SBATCH --error=logs/R40c_%A_%a.err
# R40c: is theta's representation the bottleneck? Same codes and tasks as R40, but theta reads a ONE-HOT STATE ID
# (MultiChainCodeId env, extra channels) while Q still uses only the shared code (--q_channels). Q's input and the
# episodes are identical to R40 (checked), so the R40 / R40b QRC runs are the baseline (paired seeds 1-10).
# QRC-ET split, lambda 0.9, eta 0, theta step ~0.1 per visit; lr {0.0003, 0.001, 0.003, 0.01, 0.03, 0.1}.
# 2 rewards x 5 codes x 6 learning rates x 10 seeds = 600 runs.   Summary:  python summarize_r40.py R40 R40b R40c --figure
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

PS=(0.2 0.9)
CODES=("130 1" "64 3" "32 3" "16 3" "10 5")
LRS=(0.0003 0.001 0.003 0.01 0.03 0.1)
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 1));                T=$((T / 10))
LR=${LRS[$((T % 6))]};               T=$((T / 6))
read D K <<< "${CODES[$((T % 5))]}"; T=$((T / 5))
P=${PS[$T]}
CQ=$(( (D + 99) / 100 ))                       # observation channels that hold the code (Q reads only these)
ETLR=$(python -c "print(0.1 * 100 * $CQ / 2)")  # theta: one-hot input, output = Q features (100*CQ) -> ~0.1 per visit

python algorithms/qrc-et-algo3.py --env_type minatar --env_id multichain_env:MultiChainCodeId-m32-n4-p${P}-D${D}-k${K}-v0 \
  --seed $SEED --total_timesteps 120000 --net_arch linear --no-obs_norm --q_channels $CQ --q_lr $LR --eval_values --resume \
  --et_head_name Dense_0 --et_algo3 --et_paired_rng --lamda 0.9 --et_eta 0.0 --et_algo3_target raw --no-et_algo3_lag \
  --et_split --et_theta_input extra_channels --et_lr $ETLR --exp_name R40c_et_oracle_lr${LR}
