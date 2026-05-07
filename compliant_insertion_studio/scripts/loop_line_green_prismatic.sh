#!/bin/bash
# Smoke test for line_green using the stash prismatic_peg_insertion primitive
# (which succeeded once already). Each iteration:
#   1. Release + camera-regrasp line_green (autonomous regrasp)
#   2. Hover above base1 slot via _run_hover
#   3. Run prismatic_peg_insertion to insert
#
# Usage: bash loop_line_green_prismatic.sh [N]
#   N: number of iterations (default 3)

set -uo pipefail

N=${1:-3}
OBJECT=line_green
GRASP_ID=1
SUCCESS=0

results=()

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

for ((i=1; i<=N; i++)); do
  iter_start=$(date +%s)
  log="/tmp/lg_prismatic_${i}.log"
  regrasp_log="/tmp/lg_prismatic_regrasp_${i}.log"

  echo
  echo "===== Run $i / $N — SAFE HEIGHT first (lift inserted part straight up) ====="
  # CRITICAL: when starting iter > 1 with the part inserted from prior insert,
  # the gripper is at low Z with the part seated in base1. Going to clear area
  # via place_down's horizontal trajectory would drag the inserted part
  # through the slot and rip the base apart. move_to_safe_height plans a
  # straight Z lift first.
  if ! python3 -u -m primitives.move_to_safe_height 2>&1 | tee -a "$log"; then
    echo "SAFE_HEIGHT failed; aborting."
    results+=("Run $i: SAFE_HEIGHT_FAIL")
    break
  fi

  echo "===== Run $i / $N — REGRASP ====="
  if ! python3 -u -m compliant_insertion_studio.scripts.regrasp_held_object \
       --object-name $OBJECT --grasp-id $GRASP_ID --mode real \
       --skip-camera-check 2>&1 | tee "$regrasp_log"; then
    echo "REGRASP failed; aborting."
    results+=("Run $i: REGRASP_FAIL")
    break
  fi

  current_quat=$(grep -oE '"post_regrasp_quat_xyzw": \[[^]]+\]' "$regrasp_log" | tail -1 \
                 | sed 's/.*\[//; s/\].*//; s/,/ /g')
  if [[ -z "$current_quat" ]]; then
    echo "Could not parse post-regrasp quat. Aborting."
    results+=("Run $i: PARSE_FAIL")
    break
  fi
  echo "  Captured post-regrasp quat: $current_quat"

  echo "===== Run $i / $N — ROTATE (canonicalize held orientation) ====="
  # Without this step, the regrasp's residual tilt (~7° off-axis for line_green)
  # produces a hover→insert IK trajectory that goes UP instead of DOWN. Mirrors
  # the rotate_object call in the assembly tool_sequence between place→regrasp
  # and translate_object insert.
  rotate_log="/tmp/lg_prismatic_rotate_${i}.log"
  if ! python3 -u -m primitives.rotate_object \
       --mode real \
       --object-name $OBJECT --base-name base1 \
       --current-object-orientation $current_quat 2>&1 | tee "$rotate_log"; then
    echo "ROTATE failed; skipping iteration."
    results+=("Run $i: ROTATE_FAIL")
    continue
  fi
  # Capture post-rotate held quat (rotate prints final_object_orientation)
  rotated_quat=$(grep -oE '"final_object_orientation": *\{"quat": *\{[^}]+\}' "$rotate_log" | tail -1 \
                 | grep -oE '"x": *-?[0-9.eE+-]+|"y": *-?[0-9.eE+-]+|"z": *-?[0-9.eE+-]+|"w": *-?[0-9.eE+-]+' \
                 | sed 's/.*: *//' | tr '\n' ' ')
  if [[ -n "$rotated_quat" ]]; then
    current_quat="$rotated_quat"
    echo "  Captured post-rotate quat: $current_quat"
  else
    echo "  WARN: could not parse rotated quat, using post-regrasp quat"
  fi

  echo "===== Run $i / $N — TRANSLATE_OBJECT INSERT (line_green→prismatic via line_green route) ====="
  # 2026-05-07: route through translate_object --insert so the line_green
  # branch in translate_object.py applies PER_OBJECT_BASE_OFFSET_M before
  # invoking _run_hover + prismatic_peg_insertion. Calling prismatic directly
  # would bypass the offset and target the un-corrected base position.
  python3 -u -m primitives.translate_object \
    --mode real --object-name $OBJECT --base-name base1 --grasp-id $GRASP_ID \
    --current-object-orientation $current_quat \
    --insert --insertion-type compliant \
    --use-default-base-position 2>&1 | tee -a "$log"

  iter_end=$(date +%s)
  iter_dur=$((iter_end - iter_start))

  # translate_object's line_green route emits result JSON with router tag
  # AND prismatic_peg_insertion prints "RESULT: SUCCESS"/"RESULT: FAILED".
  if grep -qE '"result": ?"success"' "$log" || grep -qE "^RESULT: SUCCESS" "$log"; then
    SUCCESS=$((SUCCESS+1))
    results+=("Run $i: OK — total ${iter_dur}s")
  else
    results+=("Run $i: FAIL — total ${iter_dur}s")
  fi

  echo
  echo "Cumulative: $SUCCESS / $i"
  sleep 2
done

echo
echo "===== SMOKE TEST COMPLETE ====="
echo "Date: $(date)"
echo "Runs: $SUCCESS / $N successful"
for r in "${results[@]}"; do
  echo "  $r"
done
