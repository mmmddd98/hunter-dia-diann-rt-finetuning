#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <status-file> <exit-code-file> <command> [args ...]" >&2
    exit 2
fi

status_file=$1
exit_code_file=$2
shift 2
printf 'running\n' >"$status_file"
"$@"
code=$?
printf '%s\n' "$code" >"$exit_code_file"
if [[ $code -eq 0 ]]; then
    printf 'completed\n' >"$status_file"
else
    printf 'failed\n' >"$status_file"
fi
exit "$code"
