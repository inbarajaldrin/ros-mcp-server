#!/usr/bin/env python3
"""Is a seated part really down? Tap its top with closed fingers and measure (2026-10-17).

    python3 -u ablations/tap_check.py [--object line_green] [--base-x X --base-y Y --base-z Z]

Operator's test: "if you are unsure you can tap ... and if it's seated then you know". Independent
of light and of the depth calibration: closed RG2 fingertips touch the part's top face; the TCP
height at contact gives the top height. Compared with the CAD top of the seated part (seat_check
TOP_FACE). Motion rules: close at safe height, go straight over the tap point at safe height,
guarded vertical descent stopping at TAP_FORCE_N and never more than MAX_PRESS_M below the
seated-top contact height, straight back up.

The tap point is the part's centre in the base frame (line_green: y = 0, between the U parts, so
nothing but the part is under the fingers). Base pose: --base-* or a fresh read (needs light).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

TAP_FORCE_N = 3.0
MAX_PRESS_M = 0.004
SEATED_TOL_MM = 1.5


def run(args, label, t=90):
    env = os.environ.copy(); env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    p = subprocess.run([sys.executable, '-u', '-m', *args], capture_output=True, text=True, timeout=t,
                       env=env, cwd=str(REPO))
    js = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith('{"result')]
    res = json.loads(js[-1]) if js else {}
    print(f">>> {label}: {res.get('result', 'rc=' + str(p.returncode))} {res.get('error', '')}"[:200], flush=True)
    return res


def tcp_z():
    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init(); n = rclpy.create_node('tap_check'); st = {}
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('p', m), 10)
    end = time.time() + 3
    while 'p' not in st and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node(); rclpy.shutdown()
    return st['p'].pose.position.z


def tap_once(obj, observed_base, offset_x_mm=0.0, offset_y_mm=0.0, force=TAP_FORCE_N,
             press_mm=1000 * MAX_PRESS_M, say=print, base='base1'):
    """One guarded tap on `obj`'s top face with closed fingers, from and back to safe height.
    Returns {'contact': bool, 'err_mm': contact height minus the seated-top contact height}."""
    from seat_check import TOP_FACE
    from primitives.shared.config import RG2_TIP_CLOSED_M, SAFE_HEIGHT
    if force > 5.0:
        raise ValueError("tap force above 5 N (operator's limit for parts in the base)")
    (bx, by, bz), bq = observed_base
    f = TOP_FACE[(obj, base)]               # the part's own CAD top face in its base (2026-10-20: was base1's centre)
    tap = np.array([bx, by, bz]) + R.from_quat(bq).apply([f['cx'] + offset_x_mm / 1000, f['cy'] + offset_y_mm / 1000,
                                                          f['z']])
    z_contact = tap[2] + RG2_TIP_CLOSED_M
    q = ['0.0', '1.0', '0.0', '0.0']
    X = lambda z: [f"{tap[0]:.4f}", f"{tap[1]:.4f}", f"{z:.4f}"]
    for args, label in [
            (['primitives.control_gripper', 'close', '--mode', 'real'], 'close'),
            (['primitives.move_to_safe_height', '--mode', 'real'], 'safe height'),
            (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(SAFE_HEIGHT + 0.03), '--quat', *q],
             'over the tap point')]:
        if run(args, label).get('result') == 'failure':
            return {'contact': False, 'err_mm': None, 'error': label}
    time.sleep(1.5)                                                    # settle before the force baseline
    res = run(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(z_contact - press_mm / 1000), '--quat', *q,
               '--min-z', '0.15', '--max-force', f"{force}", '--min-duration', '10'],
              f'tap {offset_x_mm:+.0f}/{offset_y_mm:+.0f} mm at {force:g} N', 120)
    z = tcp_z()
    run(['primitives.move_to_safe_height', '--mode', 'real'], 'straight back up')
    out = {'contact': bool(res.get('contact')), 'err_mm': round(1000 * (z - z_contact), 1)}
    say(f"  [tap] {obj} at ({offset_x_mm:+.0f}, {offset_y_mm:+.0f}) mm, {force:g} N: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', default='line_green')
    ap.add_argument('--base-x', type=float); ap.add_argument('--base-y', type=float)
    ap.add_argument('--base-z', type=float); ap.add_argument('--base-yaw-deg', type=float, default=0.0)
    ap.add_argument('--offset-x-mm', type=float, default=0.0,
                    help='tap point offset across the part (base x): press the HIGH edge of a part '
                         'cocked across its width (seat_check roll < 0 -> -x edge high)')
    ap.add_argument('--offset-y-mm', type=float, default=0.0, help='offset along the part (base y)')
    ap.add_argument('--force', type=float, default=TAP_FORCE_N, help='stop force (operator: <= 5 N)')
    ap.add_argument('--press-mm', type=float, default=1000 * MAX_PRESS_M,
                    help='deepest the tips may go below the seated top (lets a cocked part roll down)')
    a = ap.parse_args()
    from seat_check import TOP_FACE
    from primitives.shared.config import RG2_TIP_CLOSED_M, SAFE_HEIGHT
    if a.base_x is None:
        from disassemble import observe_base
        ob = observe_base('base1')
        if ob is None:
            print("base1 not observed (light?) — pass --base-x/--base-y/--base-z"); return 1
        (bx, by, bz), bq = ob
    else:
        bx, by, bz = a.base_x, a.base_y, a.base_z
        bq = R.from_euler('z', a.base_yaw_deg, degrees=True).as_quat()
    top_b = TOP_FACE[a.object]['z']                                    # top face, base frame
    if a.force > 5.0:
        print("refusing: force above 5 N (operator's limit for pressing parts in the base)"); return 1
    tap = np.array([bx, by, bz]) + R.from_quat(bq).apply([a.offset_x_mm / 1000, a.offset_y_mm / 1000, top_b])
    z_contact = tap[2] + RG2_TIP_CLOSED_M                              # flange z when tips touch
    q = ['0.0', '1.0', '0.0', '0.0']
    X = lambda z: [f"{tap[0]:.4f}", f"{tap[1]:.4f}", f"{z:.4f}"]
    print(f"[tap] {a.object}: seated top at world z {1000*tap[2]:.1f} mm -> tips touch at flange z "
          f"{1000*z_contact:.1f} mm; tap point ({1000*tap[0]:.1f}, {1000*tap[1]:.1f})", flush=True)
    for args, label in [
            (['primitives.control_gripper', 'close', '--mode', 'real'], 'close'),
            (['primitives.move_to_safe_height', '--mode', 'real'], 'safe height'),
            (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(SAFE_HEIGHT + 0.03), '--quat', *q],
             'over the tap point')]:
        if run(args, label).get('result') == 'failure':
            return 1
    time.sleep(1.5)                                                    # settle before the force baseline
    res = run(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(z_contact - a.press_mm / 1000), '--quat', *q,
               '--min-z', '0.15', '--max-force', f"{a.force}", '--min-duration', '10'], 'tap (guarded)', 120)
    z = tcp_z()
    run(['primitives.move_to_safe_height', '--mode', 'real'], 'straight back up')
    if not res.get('contact'):
        print(f"[tap] {a.object}: NO contact down to {1000*MAX_PRESS_M:.0f} mm below the seated top — part "
              f"lower than expected or not under the fingers")
        return 1
    err = 1000 * (z - z_contact)
    verdict = 'SEATED' if abs(err) <= SEATED_TOL_MM else ('HIGH' if err > 0 else 'LOW')
    print(f"[tap] {a.object}: contact at flange z {1000*z:.1f} mm, {err:+.1f} mm vs seated -> {verdict}")
    return 0 if verdict == 'SEATED' else 2


if __name__ == '__main__':
    sys.exit(main())
