#!/bin/bash
# srsClassic hub/resolver pilot: one seed per architecture (A, B, C) packed on one GPU (preempt partition, free).
#SBATCH --job-name=srs_pilot
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --constraint=a100|l40s|a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/ix/jpherman/hermanj/srs_models/logs/%x_%j.out
module load python/pytorch_251_311_cu124
cd /ix/jpherman/hermanj/srs_models/code
mkdir -p ../results
for A in A B C; do
  OMP_NUM_THREADS=1 python -m srs.train --arch $A --seed 0 --name ${A}_s0 --out ../results > ../results/${A}_s0.log 2>&1 &
done
wait
