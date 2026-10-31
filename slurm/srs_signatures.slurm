#!/bin/bash
# Signatures (Stage 1) for each trained v2 network: behaviour calibration + 10 sampled sessions each, one CPU process per
# network, 8 networks per array task, on a preempt GPU node (free). Outputs per network in sig_v2/<name>/.
#SBATCH --job-name=srs_sig
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --constraint=a100|l40s|a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/ix/jpherman/hermanj/srs_models/logs/%x_%A_%a.out
module load python/pytorch_251_311_cu124
cd /ix/jpherman/hermanj/srs_models/analysis
NETS=(A_s0 A_s1 A_s2 A_s3 A_s4 A_s5 AE_s0 AE_s1 AE_s2 AE_s3 AE_s4 AE_s5 B_s0 B_s1 B_s2 B_s3 B_s4 B_s5 C_s0 C_s4 C_s5)
for i in $(seq 0 7); do
  k=$((SLURM_ARRAY_TASK_ID * 8 + i)); [ $k -ge ${#NETS[@]} ] && break
  N=${NETS[$k]}
  [ -f sig_v2/$N/draws.csv ] && continue
  mkdir -p sig_v2/$N
  STAGE0_OUT=sig_v2/$N OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -W ignore scripts/srs_stage0.py 10 $N=/ix/jpherman/hermanj/srs_models/results2/$N/model.pt > sig_v2/$N.log 2>&1 &
done
wait
