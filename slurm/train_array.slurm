#!/bin/bash
#SBATCH --job-name=4f-ctx
#SBATCH --clusters=smp
#SBATCH --partition=smp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
# Train one model per array task. Each line of $JOBS is the argument list for
# scripts/train_context.py (without --threads).
# Usage: sbatch --array=1-N%M --export=ALL,JOBS=slurm/jobs_pilot.txt slurm/train_array.slurm

module purge
module load python/pytorch_251_311_cu124
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBS")
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${ARGS}"
python scripts/train_context.py ${ARGS} --threads ${SLURM_CPUS_PER_TASK}
