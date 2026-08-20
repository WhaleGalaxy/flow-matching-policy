#!/usr/bin/env bash
# Checklist id: w0d5t2. Downloads ManiSkill3 demonstrations for the 3 tasks
# used throughout this project. Requires `pip install mani-skill` first.
set -euo pipefail

TASKS=("PickCube-v1" "StackCube-v1" "PegInsertionSide-v1")

for t in "${TASKS[@]}"; do
  echo "==> downloading demos for $t"
  python -m mani_skill.utils.download_demo -e "$t"
done

echo "Done. Demos saved under ~/.maniskill/demos/"
