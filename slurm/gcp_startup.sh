#!/bin/bash
# GCP VM startup script (runs on every boot, so a spot VM restarted after preemption resumes).
# Fetches code + exports from the bucket, runs this VM's assigned training runs (one line of
# train_context.py arguments each) PAR at a time on the GPU, syncs results (no checkpoints) to
# the bucket every 5 min and at the end, then writes done/<name> and powers off.
# Metadata: bucket (gs://...), par (concurrent runs, default 10). Assignment: <bucket>/assign/<name>.txt
MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }
NAME=$(md name)
B=$(md attributes/bucket)
PAR=$(md attributes/par || echo 10)
W=/opt/work
exec >> /var/log/4f-startup.log 2>&1
echo "=== boot $(date) $NAME"
PY=$(command -v python3)
[ -x /opt/conda/bin/python ] && PY=/opt/conda/bin/python
if [ ! -f $W/.setup ]; then
    mkdir -p $W && cd $W
    gcloud storage cp "$B/code.tar.gz" . && tar xzf code.tar.gz
    gcloud storage cp "$B/assign/$NAME.txt" jobs.txt
    $PY -c "import h5py, scipy" 2>/dev/null || $PY -m pip install -q h5py scipy
    touch .setup
fi
cd $W
for i in $(seq 60); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
sync_results() { [ -d results ] && gcloud storage rsync -r -x '.*checkpoint\.(pt|tmp)$' results "$B/results" -q; }
status() {  # heartbeat for the local watcher: one line per run (state, last log line), GPU use
    {
        echo "time $(date -u +%FT%TZ)"
        echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null)"
        echo "procs $(pgrep -fc 'scripts/train_context.py')"
        tr -d '\r' < jobs.txt | grep -v '^$' | while read -r A; do
            O=$(echo "$A" | sed "s/.*--output-dir //; s/ .*//")
            if [ -f "$O/summary.json" ]; then S=done
            elif grep -q Traceback "$O/train.log" 2>/dev/null; then S=FAILED
            elif [ -f "$O/train.log" ]; then S=running
            else S=queued; fi
            echo "run $S $O | $(tail -n 1 "$O/train.log" 2>/dev/null | cut -c1-150)"
        done
    } | gcloud storage cp - "$B/status/$NAME.txt" -q
}
( while true; do status; sleep 120; done ) &
HB=$!
( while true; do sleep 300; sync_results; done ) &
SYNC=$!
export PY
tr -d '\r' < jobs.txt | grep -v '^$' | xargs -P "$PAR" -I{} bash -c '
    ARGS="{}"; OUT=$(echo "$ARGS" | sed "s/.*--output-dir //; s/ .*//")
    [ -f "$OUT/summary.json" ] && exit 0
    mkdir -p "$OUT"
    $PY scripts/train_context.py $ARGS --device cuda --threads 1 < /dev/null >> "$OUT/train.log" 2>&1'
kill $SYNC $HB
sync_results
status
N=$(tr -d '\r' < jobs.txt | grep -v '^$' | while read -r A; do O=$(echo "$A" | sed "s/.*--output-dir //; s/ .*//"); [ -f "$O/summary.json" ] && echo; done | wc -l)
echo "$N/$(grep -c . jobs.txt) runs finished $(date)" | gcloud storage cp - "$B/done/$NAME"
shutdown -h now
