#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1/queue.tsv" ]]; then
    echo "Usage: $0 <queue-output>" >&2
    exit 2
fi

printf '%-28s %-5s %-8s %-12s %s\n' JOB GPU PID STATUS LOG
tail -n +2 "$1/queue.tsv" | while IFS=$'\t' read -r job gpu pid status_file log_file output; do
    status=unknown
    [[ -f "$status_file" ]] && status=$(<"$status_file")
    if kill -0 "$pid" 2>/dev/null; then
        process=alive
    else
        process=stopped
    fi
    printf '%-28s %-5s %-8s %-12s %s (%s)\n' "$job" "$gpu" "$pid" "$status" "$log_file" "$process"
    if [[ -f "$output/sweep_id.txt" ]]; then
        printf '  W&B: %s\n' "$(<"$output/sweep_id.txt")"
    fi
done
