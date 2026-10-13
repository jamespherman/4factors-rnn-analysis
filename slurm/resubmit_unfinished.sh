#!/bin/bash
# Resubmit runs from a job list that have no summary.json, if none of that list's
# jobs are queued or running. Runs that wrote to train.log in the last 15 min are
# treated as running. Runs resume from their checkpoints.
# Usage (from the project dir on CRC): bash slurm/resubmit_unfinished.sh slurm/jobs_pilot.txt [PACK] [SCRIPT]
set -euo pipefail
JOBS=$1
PACK=${2:-8}
SCRIPT=${3:-scripts/train_context.py}
NAME=4f-$(basename "$JOBS" .txt)

if [ -n "$(squeue -M gpu -u "$USER" -h -n "$NAME" -o %i)" ]; then
    echo "active"
    exit 0
fi
TODO=slurm/todo_$(basename "$JOBS")
: > "$TODO"
while read -r ARGS; do
    [ -z "$ARGS" ] && continue
    OUT=""; PREV=""
    for A in $ARGS; do
        [ "$PREV" = "--output-dir" ] && OUT="$A"
        PREV="$A"
    done
    [ -f "$OUT/summary.json" ] && continue
    # Skip runs still writing their log (running under another job)
    [ -n "$(find "$OUT" -maxdepth 1 -name train.log -mmin -15 2>/dev/null)" ] && continue
    echo "$ARGS" >> "$TODO"
done < "$JOBS"
N=$(wc -l < "$TODO")
if [ "$N" -eq 0 ]; then
    echo "complete"
    exit 0
fi
NTASKS=$(( (N + PACK - 1) / PACK ))
sbatch --job-name="$NAME" --array=1-"$NTASKS" --export=ALL,JOBS="$TODO",PACK="$PACK",SCRIPT="$SCRIPT" slurm/train_gpu_pack.slurm >/dev/null
echo "resubmitted $N runs as $NTASKS tasks"
