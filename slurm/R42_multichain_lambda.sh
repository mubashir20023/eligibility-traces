#!/bin/bash
#SBATCH --job-name=R42_mc
#SBATCH --array=0-251
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --output=logs/R42_%A_%a.out
#SBATCH --error=logs/R42_%A_%a.err
# R42: full lambda sweep on the multi-chain, for QRC and QRC-ET with the SAME grids (the earlier one-hot result
# gave QRC only lambda 0 and 0.9, and learning rates a factor 3 apart).
#   12 tasks (linear agent):
#     our sparse task, one-hot            : m = 8, 32, 128 chains, reward p = 0.2 and 0.9      (as R33-R35a, plus m = 128)
#     the paper's task, one-hot  (Fig. 4) : m = 8, 32, 128; every reward +1, the last one +1 (prob 0.9) or -1
#     the paper's task, features (Fig. 5) : m = 8, 32, 128, chain length 16; branch + progress + bias + 2 flags
#   agents:
#     qrc    QRC                                           lambda {0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95}
#     et     QRC-ET, theta step 0.1 per visit (as R35a/R40; the constant step used for deep networks)
#     etcnt  QRC-ET, theta step 1/visits (running mean of the traces, as in the paper's tabular experiments)
#            both QRC-ET agents: split form, eta 0, lambda {0.2, 0.4, 0.6, 0.8, 0.9, 0.95}  (lambda 0 is the same agent as QRC)
#     etb03, etb003  QRC-ET at lambda 0.9 with theta step 0.3 and 0.03 (a check of the constant step)
#   learning rate (same for all agents) : 9 values, factor 2:  one-hot tasks 1/256 ... 1  (1 = the largest tabular step)
#                                         feature task  1/512 ... 1/2  (3 active features; long traces need small steps)
#                                         (ranges set from a one-seed pilot)
#   seeds  : 101-110 choose lambda and learning rate, 111-120 are reported (all fresh)
# One array task = one task x one agent x one lambda = 9 learning rates x 20 seeds = 180 short runs, 4 at a time.
# 252 array tasks, 45,360 runs of 120k steps.  Finished runs are skipped, so the job can simply be submitted again.
# Needs apply_count_step_patch.py (adds --et_count_step).
#   sbatch R42_multichain_lambda.sh                       QRC and QRC-ET
#   sbatch --export=ALL,BASE=q R42_multichain_lambda.sh   optional: the same with online Q(lambda) = TD(lambda) (prefix R42q)
# Summary:  python summarize_r42.py        (or: python summarize_r42.py R42q)
module -q load mamba
conda activate streamrl
cd /fred/oz511/muahmed/code/stream-rep-rl
mkdir -p logs
export JAX_PLATFORMS=cpu

BASE=${BASE:-qrc}
if [ "$BASE" = "q" ]; then PREFIX=R42q; QL="--q_lambda"; else PREFIX=R42; QL=""; fi
TASKS=(MultiChainSparse-m8-n4-p0.2 MultiChainSparse-m32-n4-p0.2 MultiChainSparse-m128-n4-p0.2
       MultiChainSparse-m8-n4-p0.9 MultiChainSparse-m32-n4-p0.9 MultiChainSparse-m128-n4-p0.9
       MultiChainPaper-m8-n4 MultiChainPaper-m32-n4 MultiChainPaper-m128-n4
       MultiChainPaperFeat-m8-n16 MultiChainPaperFeat-m32-n16 MultiChainPaperFeat-m128-n16)
ARMS=("qrc 0 0" "qrc 0.2 0" "qrc 0.4 0" "qrc 0.6 0" "qrc 0.8 0" "qrc 0.9 0" "qrc 0.95 0"
      "et 0.2 0.1" "et 0.4 0.1" "et 0.6 0.1" "et 0.8 0.1" "et 0.9 0.1" "et 0.95 0.1"
      "etcnt 0.2 0.1" "etcnt 0.4 0.1" "etcnt 0.6 0.1" "etcnt 0.8 0.1" "etcnt 0.9 0.1" "etcnt 0.95 0.1"
      "etb03 0.9 0.3" "etb003 0.9 0.03")                        # agent, lambda, constant theta step per visit
T=$SLURM_ARRAY_TASK_ID
read AGENT LAM BETA <<< "${ARMS[$((T % 21))]}"
TASK=${TASKS[$((T / 21))]}

# constant theta step per visit = BETA:  et_lr = BETA * F / (2 * K), F = input size of the linear layer, K = active features
# per state.  For etcnt the step is 1/visits instead and et_lr is not used.
M=$(echo $TASK | sed -E 's/.*-m([0-9]+)-n([0-9]+).*/\1/')
N=$(echo $TASK | sed -E 's/.*-m([0-9]+)-n([0-9]+).*/\2/')
case $TASK in
  MultiChainPaperFeat*) D=$((M + N + 3)); K=3; LRS=(0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5) ;;
  *)                    D=$((M * N + 2)); K=1; LRS=(0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25 0.5 1.0) ;;
esac
F=$(( (D + 99) / 100 * 100 ))
ETLR=$(python -c "print($BETA * $F / (2 * $K))")
CNT=""; if [ "$AGENT" = "etcnt" ]; then CNT="--et_count_step 1.0"; fi
echo "R42 task $T: $TASK $AGENT lambda $LAM (base $BASE, F $F, theta lr $ETLR $CNT)"

one() {   # one run: $1 = learning rate, $2 = seed
  local NAME=${PREFIX}_${AGENT}_lam${LAM}_lr$1
  local DIR="logs/multichain_env:${TASK}-v0__${NAME}__$2"
  [ -f "$DIR/DONE" ] && return 0
  local C="--env_type minatar --env_id multichain_env:${TASK}-v0 --seed $2 --total_timesteps 120000 --net_arch linear --no-obs_norm --q_lr $1 --lamda $LAM --eval_values --resume --exp_name $NAME $QL"
  local OUT=$(mktemp) RC=0
  if [ "$AGENT" = "qrc" ]; then
    python algorithms/qrc.py $C > "$OUT" 2>&1 || RC=$?
  else
    python algorithms/qrc-et-algo3.py $C --et_head_name Dense_0 --et_algo3 --et_paired_rng --et_eta 0.0 \
      --et_algo3_target raw --no-et_algo3_lag --et_split --et_lr $ETLR $CNT > "$OUT" 2>&1 || RC=$?
  fi
  if [ $RC -eq 0 ]; then
    touch "$DIR/DONE"; rm -f "$DIR/checkpoint.pkl" "$DIR"/*.cleanrl_model      # keep only the log and the config
  else
    echo "FAILED: $TASK $NAME seed $2"; tail -n 3 "$OUT"
  fi
  rm -f "$OUT"
}

COUNT=0
for LR in "${LRS[@]}"; do
  for SEED in $(seq 101 120); do
    one $LR $SEED &
    COUNT=$((COUNT + 1))
    if [ $((COUNT % 4)) -eq 0 ]; then wait; fi
  done
done
wait
echo "R42 task $T finished"
