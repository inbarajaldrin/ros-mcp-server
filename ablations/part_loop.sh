#!/bin/bash
# One part, N times: pull it to the clear area (camera-read seat grasp, gentle landing), reinsert it
# (held-yaw read before the insert), print the ease numbers. The per-part iteration loop of
# docs/DEBUG_LOOP.md as one command. The other parts stay where they are (seated or on the table);
# the part must be reachable (nothing seated on top of it).
#
#   bash ablations/part_loop.sh u_brown 3 [/tmp/out.log]
set -u
PART=${1:?part}; N=${2:-3}; OUT=${3:-/tmp/claude-1000/part_loop_${PART}.out}
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
: > "$OUT"
for k in $(seq 1 "$N"); do
  {
    echo "=== $PART iteration $k/$N: pull"
    python3 -u ablations/disassemble.py --assembly-json x --only "$PART" --to-staging || { echo "PULL FAILED $PART ($k)"; exit 1; }
    echo "=== $PART iteration $k/$N: reinsert"
    python3 -u ablations/replay_real_assembly.py --assembly-json "$J" --only "$PART" --skip-startup --part-at-staging || { echo "PART FAILED $PART ($k)"; exit 1; }
    echo "=== $PART iteration $k/$N: ease"
    python3 ablations/insert_ease.py --part "$PART" --last 1 2>/dev/null | tail -2
    grep -oE "\[held-yaw\] $PART: held part reads [-+0-9.]+ deg|\[verify\] $PART: OK \(marker xy [0-9.]+ mm, yaw err [-0-9.]+ deg\)" "$OUT" | tail -2
  } >> "$OUT" 2>&1 || exit 1
done
echo "LOOP DONE $PART $N" >> "$OUT"
