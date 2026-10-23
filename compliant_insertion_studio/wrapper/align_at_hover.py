#!/usr/bin/env python3
"""Align the held part to its seat in yaw at the insert hover, from the camera, before any contact (2026-10-21).

    python3 -u -m compliant_insertion_studio.wrapper.align_at_hover --object-name line_green --base-name base1 \
        [--tol-deg 0.3] [--max-iter 3] [--snapshot-dir DIR]

A stage between the hover and the descent (a funnel: measure -> correct -> re-measure until inside the tolerance).
The operator saw line_green go in turned against its slot; at the hover base1's markers sat 6-8 deg from the bar in the
image while the chained bookkeeping (base yaw, in-hand yaw, hover fix) predicted < 1 deg. This measures the error where
it matters instead of predicting it: base and held part come from the SAME still frames (/objects_poses_real, the
localizer's markers), so their RELATIVE yaw does not depend on the camera mount (a hand-eye yaw error cancels in the
difference). The target is the CAD seat orientation (assembly JSON) modulo the part's fold symmetry. The correction is
a wrist turn about world z by -error, in place (the part turns with the flange: relative yaw changes by the turn).

Prints __RESULT_JSON__ {result, iterations: [{rel_err_deg, n_base, n_part, turn_deg}], final_err_deg}. Never blocks
an insert: no measurement -> result 'unmeasured' (the caller goes on as before).
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

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]
STILL_WINDOW_S = 0.8


def still_samples(names, timeout=6.0, want=5):
    """{name: [(xyz, quat)]} of localizer poses published while the TCP was still (the observe helper's rule)."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from tf2_msgs.msg import TFMessage
    rclpy.init(); n = rclpy.create_node('align_at_hover'); pend = []; tcp = []; out = {k: [] for k in names}
    n.create_subscription(TFMessage, '/objects_poses_real', lambda m: [pend.append((time.time(), t.child_frame_id,
        (t.transform.translation.x, t.transform.translation.y, t.transform.translation.z),
        (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)))
        for t in m.transforms if t.child_frame_id in names], 10)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose',
                          lambda m: tcp.append((time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z)), 100)
    end = time.time() + timeout
    while time.time() < end and min(len(v) for v in out.values()) < want:
        rclpy.spin_once(n, timeout_sec=0.02)
        keep = []
        for it in pend:
            if not tcp or tcp[-1][0] < it[0]:
                keep.append(it); continue
            win = [c for c in tcp if it[0] - STILL_WINDOW_S <= c[0] <= it[0]]
            if len(win) >= 10 and max(np.linalg.norm(np.subtract(c[1:], win[-1][1:])) for c in win) <= 0.0005:
                out[it[1]].append((it[2], it[3]))
        pend[:] = [k for k in keep if time.time() - k[0] < 3.0]
    n.destroy_node(); rclpy.shutdown()
    return out


def mean_quat(qs):
    Q = np.array(qs); Q *= np.sign(Q @ Q[0])[:, None]; q = Q.mean(0); return q / np.linalg.norm(q)


def rel_yaw_error(obj, base, Rb, Ro, R_target_base=None):
    """Yaw (deg, about world z) taking the part's measured base-relative orientation to the nearest fold-equivalent of
    its CAD seat orientation."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
    from utils.data_path_finder import get_symmetry_dir
    _, sq = load_seat_pose_in_base(base, obj)
    # the commanded base frame when given: it carries the follow yaw (the line between the U parts' leg gaps, from
    # depth) on top of the base's own yaw - both camera-measured top-down, so a mount yaw error still cancels
    R_target_w = (R_target_base if R_target_base is not None else Rb) * R.from_quat(sq)
    try:
        fold = load_symmetry_data(obj, str(get_symmetry_dir()))
        cands = [R.from_matrix(m) for m in equivalent_orientations(R_target_w.as_matrix(), fold)]
    except Exception:
        cands = [R_target_w]
    best = min(cands, key=lambda c: (c * Ro.inv()).magnitude())
    d = (best * Ro.inv()).as_rotvec()                             # world rotation measured -> target
    return float(np.degrees(d[2])), float(np.degrees(np.hypot(d[0], d[1])))


TURN_INPROC = os.environ.get('TURN_INPROC', '1') == '1'
TURN_DEG_S, TURN_MIN_S, TURN_SHORT_DEG = 4.0, 1.0, 6.0


def turn_wrist(deg):
    """Turn the flange about world z by `deg`, in place.
    TURN_INPROC=1: on the touch insert's one node, a turn up to 6 deg timed by its angle (4 deg/s, >= 1 s) - the servo's
    1-3 deg corrections took ~4 s each as a pose-reading node + a move_cartesian process held at its 2.5 s floor
    (run 56: 16 s of still arm in 4 turns). Larger turns keep the planner's own timing."""
    if TURN_INPROC:
        from compliant_insertion_studio.wrapper.touch_insert import tcp_now, move as _tmove, close_node as _close
        try:
            p, q0 = tcp_now()
            q = (R.from_euler('z', deg, degrees=True) * R.from_quat(q0)).as_quat()
            dur = max(TURN_MIN_S, abs(deg) / TURN_DEG_S) if abs(deg) <= TURN_SHORT_DEG else None
            r = _tmove([float(v) for v in p], [float(v) for v in q], min_z=0.15, duration=dur)
        finally:
            _close()
        return r.get('result') == 'success'
    import rclpy
    from geometry_msgs.msg import PoseStamped
    rclpy.init(); n = rclpy.create_node('align_tcp'); s = {}
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: s.__setitem__('p', m.pose), 5)
    t = time.time()
    while 'p' not in s and time.time() - t < 3:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_node(); rclpy.shutdown()
    p = s['p']; q = (R.from_euler('z', deg, degrees=True)
                     * R.from_quat([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])).as_quat()
    r = subprocess.run(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                        f'{p.position.x:.4f}', f'{p.position.y:.4f}', f'{p.position.z:.4f}', '--quat',
                        *[f'{v:.6f}' for v in q], '--min-z', '0.15'], cwd=str(REPO), capture_output=True, text=True,
                       timeout=60, env=dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}"))
    return r.returncode == 0


def snapshot(out_dir, tag):
    try:
        subprocess.run([sys.executable, '-c', SNAP, out_dir, tag], timeout=20)
    except Exception:
        pass


SNAP = r"""
import sys, time, os, rclpy, numpy as np, cv2
from sensor_msgs.msg import Image
out_dir, tag = sys.argv[1], sys.argv[2]; os.makedirs(out_dir, exist_ok=True)
rclpy.init(); n = rclpy.create_node('align_snap'); g = {}
n.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: g.__setitem__('c', m), 1)
t = time.time()
while 'c' not in g and time.time() - t < 5: rclpy.spin_once(n, timeout_sec=0.1)
if 'c' in g:
    m = g['c']; a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
    if m.encoding == 'rgb8': a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(out_dir, f"align_{tag}_{time.strftime('%H%M%S')}.jpg"), a)
n.destroy_node(); rclpy.shutdown()
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--object-name', required=True); ap.add_argument('--base-name', required=True)
    ap.add_argument('--tol-deg', type=float, default=0.3); ap.add_argument('--max-iter', type=int, default=3)
    ap.add_argument('--max-turn-deg', type=float, default=12.0)
    ap.add_argument('--target-base-quat', type=float, nargs=4, default=None,
                    help='the base frame the insert aims at (with any follow yaw); default: the base as seen now')
    ap.add_argument('--snapshot-dir', default=os.environ.get('HOVER_SNAPSHOT') or None)
    a = ap.parse_args()
    its = []; final = None; result = 'unmeasured'
    for k in range(a.max_iter + 1):
        time.sleep(1.0)
        S = still_samples([a.base_name, a.object_name])
        nb, no = len(S[a.base_name]), len(S[a.object_name])
        if nb < 3 or no < 3:
            its.append({'n_base': nb, 'n_part': no, 'note': 'not enough still samples'}); break
        Rb = R.from_quat(mean_quat([q for _, q in S[a.base_name]]))
        Ro = R.from_quat(mean_quat([q for _, q in S[a.object_name]]))
        err, tilt = rel_yaw_error(a.object_name, a.base_name, Rb, Ro,
                                  R.from_quat(a.target_base_quat) if a.target_base_quat else None)
        final = err; it = {'rel_err_deg': round(err, 2), 'tilt_deg': round(tilt, 2), 'n_base': nb, 'n_part': no}
        if a.target_base_quat:                                     # commanded base (ICP + follow) vs the markers now
            it['cmd_vs_seen_base_yaw_deg'] = round(float(np.degrees(
                (R.from_quat(a.target_base_quat) * Rb.inv()).as_rotvec()[2])), 2)
            print(f"[align] commanded base vs base seen now: {it['cmd_vs_seen_base_yaw_deg']:+.2f} deg about z", flush=True)
        print(f"[align] {a.object_name}: part vs its seat {err:+.2f} deg about z (tilt {tilt:.1f}; {nb}/{no} samples)", flush=True)
        if a.snapshot_dir:
            snapshot(a.snapshot_dir, f"{a.object_name}_{k}")
        if abs(err) <= a.tol_deg:
            result = 'aligned'; its.append(it); break
        if abs(err) > a.max_turn_deg or k == a.max_iter:
            result = 'outside' if abs(err) > a.max_turn_deg else 'not_converged'; its.append(it); break
        it['turn_deg'] = round(err, 2)
        its.append(it)
        if not turn_wrist(err):                                    # turn the part by the error (world z)
            result = 'turn_failed'; break
    print('__RESULT_JSON__'); print(json.dumps({'result': result, 'object_name': a.object_name, 'iterations': its,
                                                'final_err_deg': None if final is None else round(final, 2)}))
    print('__END_RESULT_JSON__', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
