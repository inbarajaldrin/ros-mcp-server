#!/bin/bash
# One part, N times: pull it to the clear area (camera-read seat grasp, gentle landing), reinsert it
# (held-yaw read before the insert), print the ease numbers. The per-part iteration loop of
# docs/DEBUG_LOOP.md as one command. The other parts stay where they are (seated or on the table);
# the part must be reachable (nothing seated on top of it).
#
#   bash ablations/part_loop.sh u_brown 3 [/tmp/out.log]
# (no set -u: ROS setup.bash references unbound variables)
PART=${1:?part}; N=${2:-3}; OUT=${3:-/tmp/claude-1000/part_loop_${PART}.out}
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash
J=ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json
: > "$OUT"
# Parts seated ON TOP of PART (later in the assembly order and verified seated in the ledger) are pulled
# to free table spots first: 2026-10-20 the u_brown pull came down on a seated line_green (6.9 N stop).
ABOVE=$(python3 - "$PART" <<'PY'
import json, sys
order = [o['object_name'] for o in sorted(json.load(open('ablations/eval_resources/fmb1_assembly.json'))['assembly_order'], key=lambda o: o['position'])]
try: led = json.load(open('ablations/logs/assembly_state.json'))
except Exception: led = {}
part = sys.argv[1]
later = order[order.index(part) + 1:]
print(' '.join(p for p in reversed(later) if (led.get(p) or {}).get('verified')))
PY
)
for p in $ABOVE; do
  echo "=== $PART is under $p: pull $p to a free spot first" >> "$OUT"
  python3 -u ablations/disassemble.py --assembly-json x --only "$p" >> "$OUT" 2>&1 || { echo "PULL FAILED $p (clearing above $PART)" >> "$OUT"; exit 1; }
done
BAGS=${PART_LOOP_BAGS:-$HOME/Documents/fmb1_frames/part_loop}   # one diagnostic bag per iteration (2026-10-20)
mkdir -p "$BAGS"
REC=""
stop_rec() { [ -n "$REC" ] && kill -INT "$REC" 2>/dev/null && for i in $(seq 20); do kill -0 "$REC" 2>/dev/null || break; sleep 1; done; [ -n "$REC" ] && kill -0 "$REC" 2>/dev/null && kill -TERM "$REC"; REC=""; }
trap 'stop_rec' EXIT INT TERM
for k in $(seq 1 "$N"); do
  B="$BAGS/${PART}_$(date +%Y%m%d_%H%M%S)_it$k"
  DIAG=1 bash compliant_insertion_studio/scripts/record_demo_bag.sh "$B" > "$B.log" 2>&1 &
  REC=$!
  echo "=== $PART iteration $k/$N: recording $B" >> "$OUT"
  {
    echo "=== $PART iteration $k/$N: pull"
    python3 -u ablations/disassemble.py --assembly-json x --only "$PART" --to-staging || { echo "PULL FAILED $PART ($k)"; exit 1; }
    echo "=== $PART iteration $k/$N: reinsert"
    python3 -u ablations/replay_real_assembly.py --assembly-json "$J" --only "$PART" --skip-startup --part-at-staging || { echo "PART FAILED $PART ($k)"; exit 1; }
    echo "=== $PART iteration $k/$N: ease"
    python3 ablations/insert_ease.py --part "$PART" --last 1 2>/dev/null | tail -2
    grep -oE "\[held-yaw\] $PART: held part reads [-+0-9.]+ deg|\[verify\] $PART: OK \(marker xy [0-9.]+ mm, yaw err [-0-9.]+ deg\)" "$OUT" | tail -2
  } >> "$OUT" 2>&1 || exit 1
  stop_rec
done
echo "LOOP DONE $PART $N" >> "$OUT"
