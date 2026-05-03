#!/bin/bash
# Cleanly stop UR5e bringup (real or fake) + RViz + RG2 visualization helpers.
#
# Uses the proven shutdown sequence from primitives/_real_mode_stash/
# peg_in_hole_insertion.py restore_controllers() + the convention rules in
# .planning/codebase/CONVENTIONS.md (X11 safety: never SIGKILL on X11 windows;
# urscript_interface needs ~5 s SIGTERM grace before SIGKILL).
#
# Usage: close_robot.sh [-v]
#   -v : verbose (show all kills + remaining processes after each step)
#
# Exit codes:
#   0  Clean shutdown (no UR-related processes remaining)
#   1  Some processes refused to die — manual cleanup needed (script lists them)

VERBOSE=false
[[ "${1:-}" == "-v" ]] && VERBOSE=true

logv() { [[ "$VERBOSE" == "true" ]] && echo "  $*"; }
log()  { echo "$*"; }

# ----------------------------------------------------------------------
# Step 1: RViz — SIGTERM only (X11 safety per CONVENTIONS / global rules)
# ----------------------------------------------------------------------
log "[close_robot] Step 1/4: RViz (SIGTERM, X11 safety)..."
RVIZ_PIDS=$(pgrep -f "^rviz2|rviz2 -d" 2>/dev/null || true)
if [[ -n "$RVIZ_PIDS" ]]; then
    logv "RViz PIDs: $RVIZ_PIDS"
    for pid in $RVIZ_PIDS; do kill -SIGTERM "$pid" 2>/dev/null; done
    sleep 3
    for pid in $RVIZ_PIDS; do
        if ps -p "$pid" >/dev/null 2>&1; then
            log "  WARN: rviz $pid still alive after 3s SIGTERM. Sending second SIGTERM."
            kill -SIGTERM "$pid" 2>/dev/null
        fi
    done
    sleep 2
fi
log "  RViz done."

# ----------------------------------------------------------------------
# Step 2: ros2 launch — SIGINT (propagates to children)
# ----------------------------------------------------------------------
log "[close_robot] Step 2/4: ros2 launch (SIGINT)..."
LAUNCH_PIDS=$(pgrep -f "ros2.*launch.*\(ur5e\|ur_bringup\|ur5e_with_rg2\)" 2>/dev/null || true)
if [[ -n "$LAUNCH_PIDS" ]]; then
    logv "Launch PIDs: $LAUNCH_PIDS"
    for pid in $LAUNCH_PIDS; do kill -SIGINT "$pid" 2>/dev/null; done
    log "  Sent SIGINT, waiting 4s for graceful shutdown..."
    sleep 4
    # Escalate any still alive
    for pid in $LAUNCH_PIDS; do
        if ps -p "$pid" >/dev/null 2>&1; then
            log "  WARN: ros2 launch $pid didn't die from SIGINT. Sending SIGTERM."
            kill -SIGTERM "$pid" 2>/dev/null
        fi
    done
    sleep 3
    for pid in $LAUNCH_PIDS; do
        if ps -p "$pid" >/dev/null 2>&1; then
            log "  WARN: ros2 launch $pid still alive. SIGKILL (no X11)."
            kill -9 "$pid" 2>/dev/null
        fi
    done
fi
log "  ros2 launch done."

# ----------------------------------------------------------------------
# Step 3: UR driver children — SIGTERM (urscript_interface needs ~5s grace)
# ----------------------------------------------------------------------
log "[close_robot] Step 3/4: UR driver children (SIGTERM, then 5s grace, then SIGKILL)..."
PATTERNS=(
    "ros2_control_node"
    "robot_state_publisher"
    "rg2_state_publisher"
    "tool0_to_rg2"
    "static_transform_publisher"
    "controller_stopper"
    "dashboard_client"
    "urscript_interface"
    "ur_robot_state_helper"
    "trajectory_until_node"
    "controller_manager"
)
for p in "${PATTERNS[@]}"; do
    pkill -SIGTERM -f "$p" 2>/dev/null || true
done
log "  Sent SIGTERM, waiting 5s (urscript_interface SIGINT-resistance)..."
sleep 5

# Escalate stragglers
log "[close_robot] Step 4/4: Escalating stragglers (SIGKILL if still alive)..."
ALIVE_AFTER_TERM=()
for p in "${PATTERNS[@]}"; do
    PIDS=$(pgrep -f "$p" 2>/dev/null || true)
    if [[ -n "$PIDS" ]]; then
        for pid in $PIDS; do
            # Skip if it's a cursor IDE process that happens to match
            CMDLINE=$(ps -p "$pid" -o args= 2>/dev/null || true)
            if [[ "$CMDLINE" == *cursorsandbox* ]]; then continue; fi
            ALIVE_AFTER_TERM+=("$pid:$p")
            kill -9 "$pid" 2>/dev/null
        done
    fi
done
if [[ ${#ALIVE_AFTER_TERM[@]} -gt 0 ]]; then
    logv "Escalated to SIGKILL: ${ALIVE_AFTER_TERM[*]}"
fi
sleep 1

# ----------------------------------------------------------------------
# Final verification
# ----------------------------------------------------------------------
echo ""
echo "============================================================"
REMAINING=""
for p in "${PATTERNS[@]}"; do
    PIDS=$(pgrep -f "$p" 2>/dev/null || true)
    for pid in $PIDS; do
        CMDLINE=$(ps -p "$pid" -o args= 2>/dev/null || true)
        if [[ "$CMDLINE" == *cursorsandbox* ]]; then continue; fi
        REMAINING+="$pid: $CMDLINE\n"
    done
done

if [[ -z "$REMAINING" ]]; then
    echo "  [close_robot] CLEAN. No UR-related processes remaining."
    echo "============================================================"
    exit 0
else
    echo "  [close_robot] WARN: Some processes still alive after all kills:"
    echo -e "$REMAINING" | sed 's/^/    /'
    echo "  Manually inspect with:  ps -ef | grep -E 'ros2_control|ur5e|robot_state'"
    echo "============================================================"
    exit 1
fi
