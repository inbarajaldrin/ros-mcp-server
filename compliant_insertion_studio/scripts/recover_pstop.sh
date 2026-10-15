#!/bin/bash
# recover_pstop.sh — clear a PROTECTIVE_STOP and leave the arm commandable, lifted and home.
#
#   bash compliant_insertion_studio/scripts/recover_pstop.sh [--no-home]
#
# The sequence used by hand on 2026-10-14/15 (u_brown landed on, open finger on base1):
#   1. dashboard "close safety popup" + "unlock protective stop"   -> Safetystatus NORMAL
#   2. stop_run.sh: stops leftover processes, re-sends the external-control program,
#      re-activates scaled_joint_trajectory_controller
#   3. gripper safety-circuit check (a latch needs the power-cycle in CLAUDE.md)
#   4. move_to_safe_height (straight up — the arm may be resting on a part or the base),
#      then move_home (skipped with --no-home)
# The gripper is left as it is: if a part is held, release it where you choose.
#
# Refuses on EMERGENCY_STOP / FAULT / VIOLATION / SAFEGUARD_STOP: those need the pendant.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${ROBOT_IP:-192.168.1.111}"
DASH="${DASH:-$HOME/Documents/prismatic-manipulation/scripts/ur_dashboard.py}"
NO_HOME=0; [ "${1:-}" = "--no-home" ] && NO_HOME=1
cd "$REPO" || exit 1
set +u; source /opt/ros/humble/setup.bash; source ~/Desktop/ros2_ws/install/setup.bash; set -u

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$1"; exit "${2:-1}"; }
safety() { python3 "$DASH" --host "$HOST" --real safety 2>/dev/null | sed -n 's/.*Safetystatus: *//p' | tr -d ' \r'; }

S="$(safety)"
[ -n "$S" ] || die "dashboard unreachable at $HOST — check the network / pendant" 2
echo "  safety status: $S"
case "$S" in
    NORMAL|REDUCED) ok "not stopped — nothing to unlock" ;;
    PROTECTIVE_STOP)
        python3 "$DASH" --host "$HOST" --real raw "close safety popup" >/dev/null 2>&1
        sleep 1
        python3 "$DASH" --host "$HOST" --real raw "unlock protective stop" >/dev/null 2>&1
        sleep 3
        S="$(safety)"
        [ "$S" = "NORMAL" ] || die "still $S after unlock — clear it on the pendant" 3
        ok "protective stop cleared" ;;
    *) die "$S needs the pendant (not a protective stop) — nothing done" 4 ;;
esac

bash "$REPO/compliant_insertion_studio/scripts/stop_run.sh" >/tmp/recover_pstop_stoprun.log 2>&1
ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
    | grep -q "^scaled_joint_trajectory_controller.*active" \
    || die "trajectory controller not active (see /tmp/recover_pstop_stoprun.log)" 5
ok "program re-sent, trajectory controller active"

GS="$(timeout 6 ros2 topic echo --once /gripper_status 2>/dev/null | head -1)"
case "$GS" in
    *Circuit1:True*|*Circuit2:True*) die "gripper safety circuit LATCHED — power-cycle per CLAUDE.md, then re-run" 6 ;;
    "") echo "  ! no /gripper_status — is the RG2 bridge up?" ;;
    *) ok "gripper: ${GS#data: }" ;;
esac

python3 -u -m primitives.move_to_safe_height --mode real 2>&1 | grep -q '"result": "success"' \
    || die "move_to_safe_height failed — lift by hand / pendant" 7
ok "lifted to safe height"
if [ "$NO_HOME" = 0 ]; then
    python3 -u -m primitives.move_home --mode real 2>&1 | grep -q '"result": "success"' \
        || die "move_home failed" 8
    ok "home"
fi
echo "  Recovered. If a part is still held: python3 -u -m primitives.control_gripper <width> --mode real"
