#!/usr/bin/env bash
set -euo pipefail
alias_name=${1:?Usage: evaluate.sh qwen|phi|magistral dense|7|9 [tasks ...]}
run=${2:?Choose dense, 7 or 9}
shift 2
tasks=("$@")
if (( ${#tasks[@]} == 0 )); then tasks=(aime25 aime26 gpqa humaneval mbpp); fi
for seed in ${SEEDS:-0 1 2}; do
  out="results/$alias_name/$run"
  python -m layertrim.evaluate "$alias_name" "$run" "${tasks[@]}" --device "${DEVICE:-auto}" --seed "$seed" --out "$out"
  if [[ ${SCORE_CODE:-1} == 0 ]]; then continue; fi
  for task in "${tasks[@]}"; do
    case "$task" in humaneval|mbpp) bash scripts/score-code.sh "$out/seed$seed" "$task" ;; esac
  done
done
