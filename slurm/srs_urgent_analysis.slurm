#!/bin/bash
# Urgent-task analysis (scripts/srs_urgent_analysis.py) for all trained networks in results3: 8 CPU processes per
# array task, networks split by index; skips networks already analysed. Runs after the training array.
#SBATCH --job-name=srs_urgA
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --constraint=a100|l40s|a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/ix/jpherman/hermanj/srs_models/logs/%x_%A_%a.out
module load python/pytorch_251_311_cu124
cd /ix/jpherman/hermanj/srs_models/analysis3_code
NETS=($(ls -d ../results3/*/ | xargs -n1 basename))
for i in $(seq 0 7); do
  for k in $(seq $((SLURM_ARRAY_TASK_ID * 8 + i)) 24 $((${#NETS[@]} - 1))); do echo ${NETS[$k]}; done > /tmp/urg_$$_$i.txt
  ( while read N; do OMP_NUM_THREADS=1 python -W ignore scripts/srs_urgent_analysis.py ../results3 "$N"; done < /tmp/urg_$$_$i.txt ) > ../logs/urgA_${SLURM_ARRAY_TASK_ID}_$i.log 2>&1 &
done
wait
