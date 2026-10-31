#!/bin/bash
# Urgent-task training (plans/2026-10-31_urgent_task_plan.md v2): runs listed in urgent_runs.txt (one per line:
# name arch set-args...), 4 per GPU on the free preempt partition. Array index k runs lines 4k..4k+3.
#SBATCH --job-name=srs_urg
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --constraint=a100|l40s|a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/ix/jpherman/hermanj/srs_models/logs/%x_%A_%a.out
module load python/pytorch_251_311_cu124
cd /ix/jpherman/hermanj/srs_models/code3
OUT=../results3
mkdir -p $OUT
BASE="task_mode=urgent sal_gain=1.0 urgency=1.0 stage_iters=1500,1500,3000 nuisance_stages=S2,S3 train_engaged_min=0.2 train_side_bias_max=0.3 shape_vis=0.25 time_cost=0"
K=$SLURM_ARRAY_TASK_ID
sed -n "$((4*K+1)),$((4*K+4))p" urgent_runs.txt | while read N A SEED REST; do
  [ -f $OUT/$N/model.pt ] && python -c "import json,sys;l=json.load(open('$OUT/$N/log.json'));sys.exit(0 if l[-1]['iter']==6000 else 1)" && continue
  OMP_NUM_THREADS=1 python -m srs.train --name $N --arch $A --seed $SEED --out $OUT --set $BASE $REST > $OUT/$N.log 2>&1 &
done
wait
