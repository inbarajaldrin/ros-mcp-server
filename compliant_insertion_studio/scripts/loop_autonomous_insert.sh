#!/bin/bash
# Reference: derived from iter 8 working config (analysis/AUTONOMOUS_RUN_LOG.md)
#
# Run the fully-autonomous u_orange / base1 / grasp_id=1 insertion N times
# back-to-back, each with a randomized base-offset perturbation in {±7,0}mm
# variations to test convergence robustness against position uncertainty.
# Spiral R_max=8mm covers up to 7mm offset from predicted seat with ~1mm
# margin — no parameter widening needed.
#
# Peg must already be held in canonical face-down orientation (the wrapper
# retracts holding the peg between runs, so this loop chains them without
# manual setup).
#
# Usage:
#   bash compliant_insertion_studio/scripts/loop_autonomous_insert.sh [N] [flags...]
#
#   N defaults to 3 if not given.
#   --no-randomize    : use base offset (0, 0) for all runs.
#   --regrasp         : between runs, release+regrasp the part to inject natural
#                       per-grasp variation (~1-2mm). Tests algorithmic
#                       robustness without relying on synthetic --base-offset-xy.
#                       Requires camera (launch with launch_camera.sh first).
#
# Each iteration (no --regrasp):
#   1. Pick offset (random or zero) for --base-offset-xy.
#   2. Wrapper: APPROACH → SEARCH (spiral) → v4 → INSERT_DESCENT → DONE.
#   3. Cleanup: safe-height retract (peg stays held).
#
# Each iteration (with --regrasp): step 0 = release+vision-regrasp before step 1.
#
# Output:
#   /tmp/autonomous_loop_<i>.log per iteration.
#   /tmp/autonomous_regrasp_<i>.log per regrasp (if --regrasp).
#   Final pass/fail tally + summary.

set -uo pipefail

N=${1:-3}
RANDOMIZE=true
DO_REGRASP=false
OBJECT="u_orange"
GRASP_ID=""
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-randomize) RANDOMIZE=false; shift ;;
    --regrasp)      DO_REGRASP=true; shift ;;
    --object)       OBJECT="$2"; shift 2 ;;
    --grasp-id)     GRASP_ID="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 2 ;;
  esac
done

# Auto-resolve GRASP_ID from fmb1_assembly.json if not explicitly given.
if [[ -z "$GRASP_ID" ]]; then
  GRASP_ID=$(python3 -c "
from primitives.shared.config import get_grasp_id_for_assembly
g = get_grasp_id_for_assembly('$OBJECT')
print(g if g is not None else 1)
" 2>/dev/null)
  echo "Auto-resolved grasp_id=$GRASP_ID for object=$OBJECT (from fmb1_assembly.json)"
fi

HELD_QUAT="0.0296 -0.7065 0.7065 -0.0296"
SUMMARY_FILE="/tmp/autonomous_loop_summary.txt"

# Offset variations in METERS (5 cardinal/diagonal points, each ≤ 7mm magnitude
# so the spiral's R_max=8mm covers the actual hole with margin).
OFFSETS=(
  "0.007 0.000"     # +X (east)
  "-0.007 0.000"    # -X (west)
  "0.000 0.007"     # +Y (north)
  "0.000 -0.007"    # -Y (south)
  "0.005 0.005"     # +X +Y (NE diag, magnitude 7.07mm)
)
OFFSET_LABELS=("E_+7x" "W_-7x" "N_+7y" "S_-7y" "NE_diag")

success=0
total=0
declare -a results
declare -a search_durations
declare -a offset_used

cd /home/aaugus11/Documents/ros-mcp-server

start_wall=$(date +%s)
echo "===== AUTONOMOUS INSERT LOOP — N=$N ====="
echo "Start: $(date)"
echo

for i in $(seq 1 "$N"); do
  total=$((total+1))
  log="/tmp/autonomous_loop_${i}.log"
  regrasp_log="/tmp/autonomous_regrasp_${i}.log"
  iter_start=$(date +%s)

  if [[ "$RANDOMIZE" == "true" ]]; then
    idx=$((RANDOM % ${#OFFSETS[@]}))
    offset="${OFFSETS[$idx]}"
    label="${OFFSET_LABELS[$idx]}"
  else
    offset="0.000 0.000"
    label="zero"
  fi
  offset_used+=("$label")

  # Regrasp BEFORE each run (including iter 1) so the post-regrasp_quat is
  # always fresh and used by the immediately-following insert. Operator can
  # invoke this after any prior workflow (fresh-grasp pipeline, prior failed
  # insert, etc.) without worrying about quat carryover.
  current_quat="$HELD_QUAT"
  if [[ "$DO_REGRASP" == "true" ]]; then
    echo "===== Run $i / $N — REGRASP first — log: $regrasp_log ====="
    if ! python3 -u -m compliant_insertion_studio.scripts.regrasp_held_object \
         --object-name $OBJECT --grasp-id $GRASP_ID --mode real \
         --skip-camera-check \
         2>&1 | tee "$regrasp_log"; then
      echo "REGRASP failed; aborting loop."
      results+=("Run $i: REGRASP_FAIL [$label]")
      break
    fi
    new_quat=$(grep -oE '"post_regrasp_quat_xyzw": \[[^]]+\]' "$regrasp_log" | tail -1 \
               | sed 's/.*\[//;s/\].*//;s/,/ /g')
    if [[ -n "$new_quat" ]]; then
      current_quat="$new_quat"
      echo "  Captured post-regrasp quat: $current_quat"
    fi
    sleep 1
  fi

  echo "===== Run $i / $N — offset: $label ($offset) — log: $log ====="

  python3 -u -m compliant_insertion_studio.scripts.run_assembly_step \
    --object-name $OBJECT --base-name base1 --grasp-id $GRASP_ID \
    --mode real \
    --already-held \
    --current-object-orientation $current_quat \
    --base-offset-xy $offset \
    --fz -9.0 --override-fz-cap \
    --step-back auto --step-back-seconds 0.0 \
    --autonomous-search \
    --search-F-press-N 9.0 \
    --search-Fmax-N 8.0 \
    --search-v-s-mm-s 5.0 \
    --search-max-duration-s 30.0 \
    2>&1 | tee "$log"

  iter_end=$(date +%s)
  iter_dur=$((iter_end - iter_start))

  # Parse outcome from the wrapper's final summary line
  outcome=$(grep -oE 'outcome": "(success|abort)"' "$log" | tail -1 | cut -d'"' -f3)
  reason=$(grep -oE '"outcome_reason": "[^"]*"' "$log" | tail -1 | sed 's/.*"outcome_reason": "//;s/".*//' | head -c 90)
  search_dur=$(grep -oE 'after [0-9.]+s of search' "$log" | tail -1 | grep -oE '[0-9.]+')

  if [[ "$outcome" == "success" ]]; then
    success=$((success+1))
    msg="OK [$label] — search ${search_dur:-?}s, total ${iter_dur}s"
  else
    msg="FAIL [$label] ($outcome) — $reason — total ${iter_dur}s"
  fi
  results+=("Run $i: $msg")
  search_durations+=("${search_dur:-NA}")
  echo
  echo "Cumulative: $success / $total"
  echo
  sleep 2
done

end_wall=$(date +%s)
total_wall=$((end_wall - start_wall))

echo
echo "===== LOOP COMPLETE ====="
{
  echo "Date: $(date)"
  echo "Runs: $success / $total successful"
  echo "Wall time: ${total_wall}s"
  echo
  for r in "${results[@]}"; do
    echo "  $r"
  done
  echo
  echo "Search durations (s): ${search_durations[*]}"
} | tee "$SUMMARY_FILE"

# Exit code reflects whether all runs succeeded.
if (( success == total )); then
  exit 0
else
  exit 1
fi
