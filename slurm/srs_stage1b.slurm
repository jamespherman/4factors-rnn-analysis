#!/bin/bash
# srsClassic Stage 1 (v2 config: nuisance randomised in S2/S3, RT time cost, shape coding): per array task k = seed,
# train A, B, C and A with input onto E only, packed on one GPU (preempt partition, free).
#SBATCH --job-name=srs_s1b
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --constraint=a100|l40s|a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/ix/jpherman/hermanj/srs_models/logs/%x_%A_%a.out
module load python/pytorch_251_311_cu124
cd /ix/jpherman/hermanj/srs_models/code2
K=$SLURM_ARRAY_TASK_ID
OUT=../results2
mkdir -p $OUT
SET="nuisance_stages=S2,S3 train_engaged_min=0.2 train_side_bias_max=0.3 time_cost=5e-4 shape_vis=0.25"
run_named() { N=$1; shift; [ -f $OUT/$N/model.pt ] && python -c "import json,sys;l=json.load(open('$OUT/$N/log.json'));sys.exit(0 if l[-1]['iter']==4500 else 1)" && return; OMP_NUM_THREADS=1 python -m srs.train --name $N "$@" > $OUT/$N.log 2>&1; }
run_named A_s$K --arch A --seed $K --out $OUT --set $SET &
run_named B_s$K --arch B --seed $K --out $OUT --set $SET &
run_named C_s$K --arch C --seed $K --out $OUT --set $SET &
run_named AE_s$K --arch A --seed $K --out $OUT --set $SET a_input_targets=E &
wait
