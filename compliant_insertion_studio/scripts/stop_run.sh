#!/bin/bash
# stop_run.sh — stop a running assembly cleanly and leave the arm commandable.
#
#   bash compliant_insertion_studio/scripts/stop_run.sh
#
# Does NOT move the arm (it STOPS it) and does NOT stop the stack (driver/gripper/camera/grasp
# publisher keep running — that is stack_down.sh). What "clean" means, in order:
#
#   0. HALT MOTION FIRST: UR dashboard `stop` — the robot controller stops the running
#      external-control program and with it any trajectory already accepted. Killing client
#      processes does NOT stop a trajectory the controller already has (2026-10-07: the arm
#      kept moving after every process had exited; the operator had to force-stop).
#   1. SIGINT the replay orchestrator; its handler SIGINTs the active primitive's
#      process group and waits for it.
#   2. Wait for every assembly process (replay, primitives, compliant_insert wrapper)
#      to exit by itself. The insert wrapper's SIGINT handler stops force mode and
#      switches controllers back; that takes seconds, not milliseconds.
#   3. Anything still alive: SIGINT directly, then SIGTERM. Never SIGKILL — a SIGKILL
#      mid force-mode leaves nothing to switch force mode off.
#   4. Restore the arm: external-control program running (re-sent if stopped — it was
#      stopped after both mid-insert stops on 2026-10-07), every joint-claiming controller
#      (force mode, passthrough, freedrive, forward_*) off and scaled_joint_trajectory_controller
#      active, switched in one call and verified; one retry after a program re-send.
#   5. Report TCP pose and gripper width, so you know whether a part is still held.
#
# Exit: 0 stopped and commandable | 1 something survived or the arm is not commandable.

set -uo pipefail

ok()   { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }

set +u
source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/Desktop/ros2_ws/install/setup.bash" 2>/dev/null
set -u

# Patterns are assembled at runtime so this script's own command line can never match
# them (pkill/pgrep self-match killed the calling shell before — exit 144, CLAUDE.md).
# Orchestrators = every driver script under ablations/ (replay, disassemble, move_base, nudge_part, the cycle script,
# ...) and any shell running one. 2026-10-21: only the replay was stopped; disassemble.py carried on after the motion
# halt and, with a restarted run, two processes commanded the arm for ~2 min (their goals preempting each other).
R=replay_real_assembly; M=primitives; W=wrapper.compliant_insert; A=ablations
REPLAY_PAT="${A}/[a-z_0-9]+\.(py|sh)"
ASSEMBLY_PAT="(${A}/[a-z_0-9]+\.(py|sh)|-m ${M}\.|${M}/[a-z_/]+\.py|compliant_insertion_studio\.${W})"

alive() { pgrep -f "$ASSEMBLY_PAT" 2>/dev/null; }
wait_gone() {  # seconds
    local end=$((SECONDS + $1))
    while [ $SECONDS -lt $end ]; do [ -z "$(alive)" ] && return 0; sleep 0.5; done
    [ -z "$(alive)" ]
}

echo "════════════════════════════════════════════════════════════"
echo "  stop_run"
echo "════════════════════════════════════════════════════════════"

step 0/3 "Halt motion (dashboard stop)"
ROBOT_IP="${ROBOT_IP:-192.168.1.111}"
if exec 3<>"/dev/tcp/$ROBOT_IP/29999" 2>/dev/null; then
    read -r -t 3 _greet <&3
    printf 'stop\n' >&3
    if read -r -t 3 resp <&3; then ok "dashboard: $resp"; else warn "dashboard: no reply to stop"; fi
    exec 3>&- 3<&-
else
    warn "dashboard $ROBOT_IP:29999 unreachable — USE THE PENDANT / E-STOP if the arm is moving"
fi

step 1/3 "Stop the assembly processes"
if [ -z "$(alive)" ]; then
    ok "no assembly running"
else
    for pid in $(pgrep -f "$REPLAY_PAT"); do
        kill -SIGINT "$pid" 2>/dev/null && ok "SIGINT $(ps -o args= -p "$pid" 2>/dev/null | cut -c1-70) ($pid)"
    done
    if wait_gone 25; then
        ok "all assembly processes exited"
    else
        for sig in SIGINT SIGTERM; do
            pids=$(alive); [ -z "$pids" ] && break
            for pid in $pids; do
                warn "$sig $(ps -o args= -p "$pid" | cut -c1-80) ($pid)"
                kill -s "$sig" "$pid" 2>/dev/null
            done
            wait_gone 15 && break
        done
        if [ -n "$(alive)" ]; then
            warn "still alive after SIGINT+SIGTERM (not SIGKILLing):"
            ps -o pid=,args= -p "$(alive | tr '\n' ',' | sed 's/,$//')" | cut -c1-100 | sed 's/^/      /'
            exit 1
        fi
        ok "all assembly processes exited"
    fi
fi

step 2/3 "Restore a commandable arm"
ctrl_state() { timeout 8 ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' | awk -v c="$1" '$1==c{print $NF}'; }
prog_running() { timeout 5 ros2 topic echo --once /io_and_status_controller/robot_program_running 2>/dev/null | grep -q "data: true"; }

# Controllers that claim the joints and block scaled_joint_trajectory_controller. The prismatic
# insert leaves passthrough_trajectory_controller active ("Starting position together with
# trajectory_passthrough is not allowed", 2026-10-14), and force mode can come back while the
# insert's cleanup finishes. Deactivate whatever is active IN THE SAME switch that activates the
# trajectory controller, check, and retry once after re-sending the program.
CONFLICTING="force_mode_controller passthrough_trajectory_controller freedrive_mode_controller
             joint_trajectory_controller forward_position_controller forward_velocity_controller
             forward_effort_controller"
restore_controllers() {
    local active="" c
    for c in $CONFLICTING; do [ "$(ctrl_state $c)" = "active" ] && active="$active $c"; done
    if [ -n "$active" ] || [ "$(ctrl_state scaled_joint_trajectory_controller)" != "active" ]; then
        [ -n "$active" ] && echo "  deactivating:$active"
        timeout 15 ros2 control switch_controllers ${active:+--deactivate $active} \
            --activate scaled_joint_trajectory_controller >/dev/null 2>&1
    fi
    for c in $CONFLICTING; do [ "$(ctrl_state $c)" = "active" ] && return 1; done
    [ "$(ctrl_state scaled_joint_trajectory_controller)" = "active" ]
}
resend_program() {
    timeout 15 ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger "{}" >/dev/null 2>&1
    for _ in $(seq 1 10); do prog_running && return 0; sleep 1; done
    return 1
}

if prog_running; then
    ok "external-control program running"
else
    resend_program && ok "external-control program re-sent and running" \
        || { warn "external-control program NOT running — check the pendant (Remote Control? protective stop?)"; exit 1; }
fi

if restore_controllers; then
    ok "scaled_joint_trajectory_controller active, force/passthrough modes off"
else
    sleep 2                                         # an insert cleanup may still be switching
    resend_program >/dev/null
    if restore_controllers; then
        ok "scaled_joint_trajectory_controller active (after program re-send)"
    else
        warn "arm NOT commandable: $(timeout 8 ros2 control list_controllers 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g' \
              | awk '$NF=="active"{printf "%s ", $1}')"
        exit 1
    fi
fi

step 3/3 "Where the arm is"
POSE="$(timeout 5 ros2 topic echo --once /tcp_pose_broadcaster/pose 2>/dev/null)"
echo "$POSE" | awk '/position:/{p=1} p&&/[xyz]:/{printf "%s%.1f mm ", $1, $2*1000} /orientation:/{exit}' | sed 's/^/  TCP /'; echo
W_MM="$(timeout 5 ros2 topic echo --once /gripper_width 2>/dev/null | awk '/data:/{print $2; exit}')"
[ -n "$W_MM" ] && echo "  gripper width ${W_MM} mm (a part is likely held if it is well below the open width)"
echo
echo "  Arm is stopped and commandable. Nothing was moved. If a part is held:"
echo "    python3 -u -m primitives.control_gripper <width> --mode real   # release"
echo "    python3 -u -m primitives.move_to_safe_height --mode real       # lift"
echo "    python3 -u -m primitives.move_home --mode real                 # home"
exit 0
