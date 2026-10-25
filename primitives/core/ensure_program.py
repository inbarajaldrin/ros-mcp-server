#!/usr/bin/env python3
"""Make sure the UR external-control program is running; re-send it and re-activate the trajectory controller if not.

2026-10-20: the driver logged "Connection to reverse interface dropped" right after a successful move (no safety
event); every later move was "goal rejected" / "External control program stopped". The documented manual recovery
(CLAUDE.md) is: resend_robot_program, then activate scaled_joint_trajectory_controller. Refuses when the safety
status is not NORMAL (a protective stop is for a person to look at first).

    python3 -m primitives.core.ensure_program
Prints __RESULT_JSON__ {result, was_running, recovered}.
"""
import json
import subprocess
import sys
import time


def result(d):
    print("__RESULT_JSON__"); print(json.dumps(d)); print("__END_RESULT_JSON__", flush=True)


def program_running(timeout=4.0):
    import rclpy
    from std_msgs.msg import Bool
    rclpy.init(); n = rclpy.create_node('ensure_program'); st = {}
    n.create_subscription(Bool, '/io_and_status_controller/robot_program_running', lambda m: st.__setitem__('r', m.data), 1)
    end = time.time() + timeout
    while 'r' not in st and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node(); rclpy.shutdown()
    return st.get('r')


def safety_normal():
    r = subprocess.run(['python3', 'scripts/ur_dashboard.py', '--host', '192.168.1.111', '--real', 'safety'],
                       cwd='/home/aldrin/Documents/prismatic-manipulation', capture_output=True, text=True, timeout=20)
    return 'NORMAL' in r.stdout


def main():
    if program_running():
        result({'result': 'success', 'was_running': True, 'recovered': False}); return 0
    if not safety_normal():
        result({'result': 'failure', 'was_running': False, 'error': 'safety status not NORMAL - not re-sending'}); return 1
    ok = False
    # 10-23 02:21: one re-send + 13 s of polling gave up; the driver reported the program back 28 s after the drop -
    # re-send up to three times, polling ~10 s after each
    for _attempt in range(3):
        subprocess.run(['ros2', 'service', 'call', '/io_and_status_controller/resend_robot_program', 'std_srvs/srv/Trigger',
                        '{}'], capture_output=True, text=True, timeout=20)
        time.sleep(3.0)
        # a real restart, as stop_run.sh does: an 'activate' on a controller the manager still lists active is a no-op
        # ('not inactive ... Empty activate and deactivate list'), the controller writes nothing, and the program drops
        # again ~30 s later - 10-24 11:13 and 18:28 looped drop -> resend -> drop until stop_run.sh cycled it
        subprocess.run(['ros2', 'control', 'switch_controllers', '--deactivate', 'scaled_joint_trajectory_controller',
                        'force_mode_controller', 'passthrough_trajectory_controller'],
                       capture_output=True, text=True, timeout=20)
        subprocess.run(['ros2', 'control', 'switch_controllers', '--activate', 'scaled_joint_trajectory_controller'],
                       capture_output=True, text=True, timeout=20)
        for _ in range(4):                      # the program reports running a few seconds after the re-send
            time.sleep(2.0)
            if program_running():
                ok = True; break
        if ok:
            break
    result({'result': 'success' if ok else 'failure', 'was_running': False, 'recovered': ok})
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
