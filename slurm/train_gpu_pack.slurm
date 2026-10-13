#!/bin/bash
#SBATCH --job-name=4f-ctx
#SBATCH --clusters=gpu
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=slurm/logs/%x_%A_%a.out
# Run PACK training runs concurrently on one GPU. Array task i runs lines
# (i-1)*PACK+1 .. i*PACK of $JOBS (arguments for scripts/train_context.py).
# preempt bills nothing to the allocation; preempted jobs are requeued and each
# run resumes from its checkpoint (finished runs are skipped).
# Usage: sbatch --array=1-N --export=ALL,JOBS=slurm/jobs_pilot.txt,PACK=8 slurm/train_gpu_pack.slurm

module purge
module load python/pytorch_251_311_cu124
PACK=${PACK:-8}
FIRST=$(( (SLURM_ARRAY_TASK_ID - 1) * PACK + 1 ))
LAST=$(( FIRST + PACK - 1 ))
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname), lines ${FIRST}-${LAST}, restart ${SLURM_RESTART_COUNT:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mapfile -t LINES < <(sed -n "${FIRST},${LAST}p" "$JOBS")
for ARGS in "${LINES[@]}"; do
    [ -z "$ARGS" ] && continue
    OUT=""; PREV=""
    for A in $ARGS; do
        [ "$PREV" = "--output-dir" ] && OUT="$A"
        PREV="$A"
    done
    mkdir -p "$OUT"
    python scripts/train_context.py ${ARGS} --device cuda --threads 1 < /dev/null >> "${OUT}/train.log" 2>&1 &
done
wait
echo "Task ${SLURM_ARRAY_TASK_ID} finished"
