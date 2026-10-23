#!/bin/bash
# assembly_cycle.sh ASSEMBLY_JSON LABEL — one M2 cycle of any assembly (2026-10-20): pull every part to a free
# spot in the cell layout's staging region (disassemble.py), then reassemble it (replay_real_assembly.py),
# one DIAG bag for the cycle. Stops at the first failure (exit code of the failed stage). Base read: the 6-dof
# CAD fit (BASE_FROM=icp, falls back to the outline, then the markers).
#   bash ablations/assembly_cycle.sh ablations/ground_truth_resources/Assembly_fmb_assembly_3_results.json c1
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash
J=$1; L=${2:-cycle}
TS=$(date +%Y%m%d_%H%M%S); B=~/Documents/fmb1_frames/cycles/${L}_$TS; mkdir -p ~/Documents/fmb1_frames/cycles
REC=""
stop_rec() { [ -n "$REC" ] && kill -INT "$REC" 2>/dev/null && for i in $(seq 20); do kill -0 "$REC" 2>/dev/null || break; sleep 1; done; [ -n "$REC" ] && kill -0 "$REC" 2>/dev/null && kill -TERM "$REC"; REC=""; }
trap 'stop_rec' EXIT INT TERM
DIAG=1 bash compliant_insertion_studio/scripts/record_demo_bag.sh "$B" > "$B.log" 2>&1 &
REC=$!
export BASE_FROM=${BASE_FROM:-icp} NO_AIM_BIAS=1
echo "=== CYCLE $L: disassemble"
python3 -u ablations/disassemble.py --assembly-json "$J"; RC=$?
if [ $RC -ne 0 ]; then echo "CYCLE EXIT $L stage=disassemble rc=$RC bag=$B"; exit $RC; fi
if [ -n "$MOVE_BASE" ]; then          # e.g. MOVE_BASE="--dy 30" or "--dyaw 5": the arm moves the empty base
  BN=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_name'])" "$J")
  echo "=== CYCLE $L: move the base ($MOVE_BASE)"
  python3 -u ablations/move_base.py --base "$BN" $MOVE_BASE; RC=$?
  if [ $RC -ne 0 ]; then echo "CYCLE EXIT $L stage=move_base rc=$RC bag=$B"; exit $RC; fi
fi
echo "=== CYCLE $L: assemble"
python3 -u ablations/replay_real_assembly.py --assembly-json "$J"; RC=$?
stop_rec
echo "CYCLE EXIT $L stage=assemble rc=$RC bag=$B"
exit $RC
