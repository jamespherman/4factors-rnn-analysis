#!/bin/bash
# GCP spot-GPU fleet for train_context.py job lists (see slurm/gcp_startup.sh).
#   bash slurm/gcp_fleet.sh launch <todo.txt> <n_vms> <zone1,zone2,...>   split runs over n VMs, create them
#   bash slurm/gcp_fleet.sh watch                restart preempted VMs, delete finished ones, until none left
#   bash slurm/gcp_fleet.sh fetch                copy results (no checkpoints) from the bucket into results/
# VMs are named fourf-gpu-NN; each gets assign/<name>.txt in the bucket.
PROJECT=${PROJECT:-herman-rvit-pitt}
BUCKET=${BUCKET:-gs://herman-4factors-rnn-v13}
MACHINE=${MACHINE:-a2-highgpu-1g}
IMAGE=pytorch-2-9-cu129-ubuntu-2204-nvidia-580
PAR=${PAR:-10}
G="gcloud --project $PROJECT --quiet"
quiet() { grep -v "FutureWarning\|warnings.warn\|environment' tag\|resource-manager\|designate_project"; }

create() {  # name, comma-separated zones: first zone with capacity wins
    local name=$1 z
    for z in ${2//,/ }; do
        $G compute instances create "$name" --zone "$z" --machine-type "$MACHINE" \
            --provisioning-model SPOT --instance-termination-action STOP --maintenance-policy TERMINATE \
            --image-family $IMAGE --image-project deeplearning-platform-release \
            --boot-disk-size 100GB --scopes cloud-platform \
            --metadata "bucket=$BUCKET,par=$PAR,install-nvidia-driver=True" \
            --metadata-from-file startup-script=slurm/gcp_startup.sh 2>&1 | quiet | grep -q RUNNING \
            && { echo "$name $z"; return 0; }
    done
    echo "$name: no capacity in $2"; return 1
}

case $1 in
launch)
    TODO=$2; N=$3; ZONES=$4; TMP=$(mktemp -d)
    tr -d '\r' < "$TODO" | grep -v '^$' > "$TMP/all.txt"
    split -n r/"$N" -d -a 2 "$TMP/all.txt" "$TMP/part"   # round robin, so sessions/variants mix
    for f in "$TMP"/part*; do
        name=fourf-gpu-$(( 10#${f##*part} + ${OFFSET:-1} ))
        name=fourf-gpu-$(printf %02d ${name##*-})
        gcloud storage cp "$f" "$BUCKET/assign/$name.txt" 2>&1 | quiet | grep -i error
        create "$name" "$ZONES" &
        sleep 2
    done
    wait ;;
watch)
    # Every 5 min: restart preempted VMs, delete finished ones, read heartbeats (status/<name>.txt).
    # Prints ALERT lines for failed runs, stale heartbeats (VM reset) and no progress for 60 min.
    RATE=${RATE:-1.6}; VMMIN=${VMMIN:-0}   # assumed $/VM-hour (A100 spot + disk), for the running cost estimate
    LASTDONE=-1; LASTCHANGE=$(date +%s); declare -A SEEN
    while true; do
        LIST=$($G compute instances list --filter="name~^fourf-gpu" --format="csv[no-heading](name,zone.basename(),status)" 2>/dev/null | tr -d '\r')
        [ -z "$LIST" ] && { echo "$(date +%H:%M) COMPLETE: no VMs left; est. cost \$$(( VMMIN * ${RATE/./} / 600 ))"; exit 0; }
        DONE=$(gcloud storage ls "$BUCKET/done/" 2>/dev/null | tr -d '\r' | sed 's#.*/##')
        NRUN=$(echo "$LIST" | grep -c ',RUNNING$'); VMMIN=$(( VMMIN + 5 * NRUN ))
        TMP=$(mktemp -d); gcloud storage cp "$BUCKET/status/*" "$TMP/" -q 2>/dev/null
        cat "$TMP"/*.txt 2>/dev/null | grep '^run ' | awk '{print $2}' | sort | uniq -c > "$TMP/counts"
        NDONE=$(awk '$2=="done"{print $1}' "$TMP/counts"); NDONE=${NDONE:-0}
        echo "$(date +%H:%M) VMs: $(echo "$LIST" | cut -d, -f3 | sort | uniq -c | tr -s ' \n' ' ')| runs: $(tr -s ' \n' ' ' < "$TMP/counts")| finished VMs $(echo "$DONE" | grep -c .) | est. cost \$$(( VMMIN * ${RATE/./} / 600 ))"
        for f in "$TMP"/*.txt; do
            [ -f "$f" ] || continue
            grep '^run FAILED' "$f" | awk '{print $3}' | while read -r o; do
                [ -z "${SEEN[$o]}" ] && echo "  ALERT failed run $o ($(basename "$f" .txt))"; done
        done
        while IFS=, read -r name zone status; do
            if echo "$DONE" | grep -qx "$name"; then
                [ "$status" != RUNNING ] && $G compute instances delete "$name" --zone "$zone" 2>&1 | quiet >/dev/null && echo "  deleted $name (finished: $(gcloud storage cat "$BUCKET/done/$name" 2>/dev/null))"
            elif [ "$status" = TERMINATED ]; then
                $G compute instances start "$name" --zone "$zone" 2>&1 | quiet | grep -qi error && echo "  $name: restart failed (no spot capacity?), will retry" || echo "  restarted preempted $name"
            elif [ "$status" = RUNNING ] && [ -f "$TMP/$name.txt" ]; then
                AGE=$(( $(date +%s) - $(date -d "$(sed -n 's/^time //p' "$TMP/$name.txt")" +%s) ))
                if [ "$AGE" -gt 1200 ]; then
                    echo "  ALERT $name heartbeat ${AGE}s old; resetting"
                    $G compute instances reset "$name" --zone "$zone" 2>&1 | quiet >/dev/null
                fi
            fi
        done <<< "$LIST"
        for f in "$TMP"/*.txt; do grep '^run FAILED' "$f" 2>/dev/null | awk '{print $3}'; done | while read -r o; do echo "$o"; done > "$TMP/failed"
        while read -r o; do SEEN[$o]=1; done < "$TMP/failed"
        if [ "$NDONE" -gt "$LASTDONE" ]; then LASTDONE=$NDONE; LASTCHANGE=$(date +%s)
        elif [ $(( $(date +%s) - LASTCHANGE )) -gt 3600 ] && [ "$NRUN" -gt 0 ]; then
            echo "  ALERT no run finished in the last $(( ($(date +%s) - LASTCHANGE) / 60 )) min"; LASTCHANGE=$(date +%s)
        fi
        rm -rf "$TMP"
        sleep 300
    done ;;
fetch)
    gcloud storage rsync -r "$BUCKET/results" results 2>&1 | quiet | tail -2 ;;
esac
