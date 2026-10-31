#!/bin/bash
#SBATCH --job-name=R31_theta
#SBATCH --array=0-239
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R31_theta_%A_%a.out
#SBATCH --error=logs/R31_theta_%A_%a.err
# R31: can Theta learn the trace? ET (eta 0) on the multi-chain PREDICTION task, same seeds as R30.
# 4 envs (m = 1, 32 x sigma = 0, 1) x 2 Theta targets (e_w = Algorithm 3 recursive, raw = ordinary trace)
# x 3 Theta learning rates (0.03, 0.1, 0.3) x 2 Q learning rates (3e-3, 1e-2; R30 showed ET is best at the top of its range) x 5 seeds = 240 tasks.
# Compare against R30's best QRC on the same envs/seeds (summarize_r31.py does this).

module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

ENVS=(m1-n4-s0.0 m32-n4-s0.0 m1-n4-s1.0 m32-n4-s1.0)
TARGETS=(e_w raw)
ETLRS=(0.03 0.1 0.3)
QLRS=(3e-3 1e-2)
SEEDS=(1 2 3 4 5)

T=$SLURM_ARRAY_TASK_ID
SEED=${SEEDS[$((T % 5))]};      T=$((T / 5))
QLR=${QLRS[$((T % 2))]};        T=$((T / 2))
ETLR=${ETLRS[$((T % 3))]};      T=$((T / 3))
TGT=${TARGETS[$((T % 2))]};     T=$((T / 2))
ENV=${ENVS[$T]}

python algorithms/qrc-et-algo3.py --env_type minatar --env_id multichain_env:MultiChainPred-${ENV}-v0 \
  --seed $SEED --total_timesteps 100000 --q_lr $QLR --lamda 0.8 --eval_values --resume \
  --et_algo3 --et_paired_rng --et_eta 0.0 --et_algo3_target $TGT --et_lr $ETLR \
  --exp_name R31_et_${TGT}_etlr${ETLR}_lr${QLR}
