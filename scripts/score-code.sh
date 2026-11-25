#!/usr/bin/env bash
set -euo pipefail
folder=$(cd "${1:?Usage: score-code.sh RESULTS/seed0 humaneval|mbpp}" && pwd)
task=${2:?Choose humaneval or mbpp}
case "$task" in
  humaneval) data=humaneval ;;
  mbpp) data=mbppplus ;;
  *) exit 2 ;;
esac
gen="$folder/generations_$task.json"
dataset="$(pwd)/data/$data.json"
test -f "$gen" && test -f "$dataset" || exit 1
tmp=$(mktemp "$folder/.score.XXXXXX")
trap 'rm -f "$tmp"' EXIT
python3 -c 'import json, sys
for name in sys.argv[1:]:
    with open(name) as f:
        print(json.dumps(json.load(f)))' "$gen" "$dataset" | \
docker run -i --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --user 65534:65534 --memory 4g --cpus 4 --pids-limit 128 --tmpfs /tmp:rw,size=256m \
  layertrim-scorer "$task" > "$tmp"
mv "$tmp" "$folder/score_$task.json"
