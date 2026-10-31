#!/bin/bash
# srsClassic Stage 1 ensemble: per array task k (seed), train A, B, C and A with input onto E only, packed on one GPU.
#SBATCH --job-name=srs_s1
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --constraint=a100|l40s|a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=1-4
#SBATCH --output=/ix/jpherman/hermanj/srs_models/logs/%x_%A_%a.out
module load python/pytorch_251_311_cu124
cd /ix/jpherman/hermanj/srs_models/code
K=$SLURM_ARRAY_TASK_ID
run_named() { N=$1; shift; [ -f ../results/$N/model.pt ] && python -c "import json,sys;l=json.load(open('../results/$N/log.json'));sys.exit(0 if l[-1]['iter']==4500 else 1)" && return; OMP_NUM_THREADS=1 python -m srs.train --name $N "$@" > ../results/$N.log 2>&1; }
run_named A_s$K --arch A --seed $K --out ../results &
run_named B_s$K --arch B --seed $K --out ../results &
run_named C_s$K --arch C --seed $K --out ../results &
run_named AE_s$K --arch A --seed $K --out ../results --set a_input_targets=E &
wait
