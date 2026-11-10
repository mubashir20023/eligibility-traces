#!/bin/bash
#SBATCH --job-name=R38_final
#SBATCH --array=0-99
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R38F_%A_%a.out
#SBATCH --error=logs/R38F_%A_%a.err
# R38 (stage 2): best QRC and best QRC-ET setting per game from the R38 tuning, 10 fresh seeds (401-410).
#   sbatch --export=ALL,ARCH=linear R38_nonconv_final.sh   (needs r38lin_best.txt)
#   sbatch --export=ALL,ARCH=mlp    R38_nonconv_final.sh   (needs r38mlp_best.txt)
# Summary:  python summarize_r34c.py R38linF   |   python summarize_r34c.py R38mlpF
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

case $ARCH in
  linear) PRE=R38lin2; HEAD="--et_head_name Dense_0" ;;
  mlp)    PRE=R38mlp; HEAD="" ;;
  *) echo "set ARCH=linear or ARCH=mlp"; exit 1 ;;
esac
[ "$ARCH" = "linear" ] && NORM="--no-obs_norm" || NORM=""
T=$SLURM_ARRAY_TASK_ID
SEED=$((T % 10 + 401))
read GAME METHOD LAM R LR <<< "$(sed -n "$((T / 10 + 1))p" ${PRE,,}_best.txt)"
C="--env_type minatar --env_id MinAtar/${GAME}-v1 --seed $SEED --total_timesteps 5000000 --net_arch $ARCH $NORM --q_lr $LR --lamda $LAM --resume"
if [ "$METHOD" = "qrc" ]; then
  python algorithms/qrc.py $C --exp_name ${PRE}F_qrc_best_lr${LR}
else
  ETLR=$(python -c "print($R * $LR)")
  python algorithms/qrc-et-algo3.py $C $HEAD --et_lr $ETLR --et_algo3 --et_paired_rng --et_algo3_target raw \
    --no-et_algo3_lag --et_split --et_eta 0.3 --exp_name ${PRE}F_et_best_lr${LR}
fi
