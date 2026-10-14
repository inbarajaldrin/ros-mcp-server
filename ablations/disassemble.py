#!/usr/bin/env python3
"""Robot self-disassembly of a seated assembly onto free table spots (2026-10-14).

    python3 -u ablations/disassemble.py --assembly-json <fmb1 replay json> [--dry-run]

Reverse assembly order. Each seated part is taken with the SAME grasp the insert left it in:
the flange pose translate_object/compliant_insert aim at (predict_tcp_at_seat: CAD seat in the
camera-observed base + the part's grasp point + per-object z offset, top-down at yaw 180 like
every logged insert) - not move_to_grasp on the seated part, whose grasp-point read put the
fingers 34 mm too deep beside line_green (2026-10-14, -71 N). Per part: hover 10 cm above that
pose -> open to the release width -> slow straight descent with a force guard -> close ->
straight pull to safe height -> move_cartesian above a free spot -> move_down (glitch-proof
contact) -> release -> safe height. Free spots come from a depth snapshot taken from home:
table height map, flat clear squares, away from the base and the staging area, reachable and
in the home view (so the next assembly can see every part). Fails closed: the first failed step
stops with the arm still, and the remaining parts stay where they are.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scipy.spatial.transform import Rotation as R  # noqa: E402

X_CAM = np.eye(4)   # localizer's calibrated camera mount (config/robot_config.yaml, 2026-10-13)
X_CAM[:3, :3] = R.from_quat([-0.0238, -0.00386, -0.01645, 0.99957]).as_matrix()
X_CAM[:3, 3] = [-0.01446, -0.04424, 0.07317]
STAGING_XY = (-0.320, -0.500)
SPOT_SIZE = 0.10                 # free square per part (m)
TABLE_TOL = 0.006                # "flat" = within this of the table plane (paper board counts)


def env():
    e = os.environ.copy()
    e['PYTHONPATH'] = f"{REPO}:{e.get('PYTHONPATH', '')}"
    return e


def run(argv, label, timeout=180):
    print(f">>> {label}: {' '.join(argv[3:])}", flush=True)
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env(), cwd=str(REPO))
    out = p.stdout + p.stderr
    js = [l for l in out.splitlines() if l.startswith('{"result"')]
    res = json.loads(js[-1]) if js else {}
    ok = p.returncode == 0 and res.get('result', 'success') != 'failure'
    print(f"    {'ok' if ok else 'FAILED'} {res.get('error', '') if not ok else ''}", flush=True)
    if not ok:
        print('\n'.join(out.splitlines()[-15:]))
    return ok, res


def py(*a):
    return [sys.executable, '-u', '-m', *a]


def depth_free_spots(n_needed, exclude_xy):
    """Free table spots from one aligned-depth frame taken where the arm is (home)."""
    import rclpy
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    rclpy.init(); n = rclpy.create_node('disassemble_depth'); st = {}
    n.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: st.__setitem__('d', m), 2)
    n.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 2)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('t', m), 10)
    end = time.time() + 8
    while time.time() < end and not all(k in st for k in 'dit'):
        rclpy.spin_once(n, timeout_sec=0.05)
    if not all(k in st for k in 'dit'):
        n.destroy_node(); rclpy.shutdown(); return None, "no depth/camera_info/tcp"
    d = CvBridge().imgmsg_to_cv2(st['d'], 'passthrough').astype(float) / 1000.0
    K = np.array(st['i'].k).reshape(3, 3); p, o = st['t'].pose.position, st['t'].pose.orientation
    n.destroy_node(); rclpy.shutdown()
    T = np.eye(4); T[:3, :3] = R.from_quat([o.x, o.y, o.z, o.w]).as_matrix(); T[:3, 3] = [p.x, p.y, p.z]
    Tc = T @ X_CAM; h, w = d.shape; uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    v = d > 0.1
    Pc = np.stack([(uu - K[0, 2]) / K[0, 0] * d, (vv - K[1, 2]) / K[1, 1] * d, d], -1)[v]
    Pw = Pc @ Tc[:3, :3].T + Tc[:3, 3]
    table_z = np.percentile(Pw[:, 2], 30)                      # most of the view is bare table
    cell = 0.01; x0, y0 = Pw[:, 0].min(), Pw[:, 1].min()
    nx, ny = int((Pw[:, 0].max() - x0) / cell) + 1, int((Pw[:, 1].max() - y0) / cell) + 1
    seen = np.zeros((nx, ny), bool); high = np.zeros((nx, ny), bool)
    ix = ((Pw[:, 0] - x0) / cell).astype(int); iy = ((Pw[:, 1] - y0) / cell).astype(int)
    seen[ix, iy] = True; high[ix[Pw[:, 2] > table_z + TABLE_TOL], iy[Pw[:, 2] > table_z + TABLE_TOL]] = True
    k = int(SPOT_SIZE / cell) + 2                              # spot + clearance, in cells
    def boxsum(m):                                             # sum over every k x k window (summed-area table)
        S = np.pad(m.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]
    ok = (boxsum(seen) > 0.9 * k * k) & (boxsum(high) == 0)
    cands = []
    for i, j in zip(*np.nonzero(ok)):
                cx, cy = x0 + (i + k / 2) * cell, y0 + (j + k / 2) * cell
                r = np.hypot(cx, cy)
                if not (0.30 < r < 0.68):
                    continue
                if any(np.hypot(cx - ex, cy - ey) < er for ex, ey, er in exclude_xy):
                    continue
                cands.append((cx, cy))
    spots = []
    for c in sorted(cands, key=lambda c: np.hypot(c[0] - 0.05, c[1] + 0.40)):   # prefer near the home view centre
        if all(np.hypot(c[0] - s[0], c[1] - s[1]) > SPOT_SIZE + 0.01 for s in spots):
            spots.append(c)
        if len(spots) == n_needed:
            break
    depth_free_spots.last = (Tc, K, table_z)
    return spots, f"table z {1000*table_z:.1f} mm, {len(cands)} candidate cells"


SEAT_EE_QUAT = (0.0, 1.0, 0.0, 0.0)        # top-down, yaw 180: every logged insert (2026-10-07/14)
HOVER_DZ = 0.10


def seat_grasp(name, gid, observed_base, base_name='base1'):
    """Flange pose the insert holds `name` at when seated (same chain as translate_object)."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base, predict_tcp_at_seat
    from primitives.shared.config import get_object_base_offset_m
    (bx, by, bz), bq = observed_base
    dz = float(get_object_base_offset_m(name)[2])
    _, seat_q = load_seat_pose_in_base(base_name, name)
    held = (R.from_quat(bq) * R.from_quat(seat_q)).as_quat()       # seated orientation in world
    out = predict_tcp_at_seat(base_name=base_name, object_name=name, grasp_id=gid,
                              base_world_xyz=[bx, by, bz + dz], base_world_quat_xyzw=bq,
                              held_quat_xyzw=held, ee_orientation_xyzw=SEAT_EE_QUAT)
    return np.array(out['predicted_tcp_at_seat']['xyz_m'])


def observe_base(base_name='base1'):
    """Read the base from straight above (as the replay's startup), then stay there."""
    from primitives.shared.config import DEFAULT_BASE_POSITION as DBP
    sys.path.insert(0, str(REPO / 'ablations'))
    from replay_real_assembly import observe_base_pose
    ok, _ = run(py('primitives.move_cartesian', '--mode', 'real', '--xyz', f"{DBP[0]:.4f}", f"{DBP[1]:.4f}", "0.34",
                   '--quat', *map(str, SEAT_EE_QUAT)), 'base view', 60)
    if not ok:
        return None
    return observe_base_pose(base_name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--assembly-json', required=True)
    ap.add_argument('--dry-run', action='store_true', help='plan spots and print steps; no motion')
    ap.add_argument('--only', default=None)
    a = ap.parse_args()
    order = json.load(open(REPO / 'ablations/eval_resources/fmb1_assembly.json'))['assembly_order']
    parts = [o for o in sorted(order, key=lambda o: -o['position']) if not a.only or o['object_name'] == a.only]
    from primitives.shared.config import DEFAULT_BASE_POSITION
    if not a.dry_run:
        ok, _ = run(py('primitives.move_home', '--mode', 'real'), 'home', 60)
        if not ok: return 1
        ok, _ = run(py('primitives.control_gripper', 'open', '--mode', 'real'), 'open', 30)
        if not ok: return 1
    exclude = [(DEFAULT_BASE_POSITION[0], DEFAULT_BASE_POSITION[1], 0.15), (*STAGING_XY, 0.15)]
    spots, note = depth_free_spots(len(parts), exclude)
    print(f"[spots] {note}: {[(round(1000*x), round(1000*y)) for x, y in (spots or [])]} mm")
    if not spots or len(spots) < len(parts):
        print("not enough free spots seen from here — stopping")
        if not a.dry_run: return 1
    ob = observe_base() if not a.dry_run else None
    if not a.dry_run and ob is None:
        print("base1 not observed — stopping"); return 1
    if a.dry_run:
        sys.path.insert(0, str(REPO / 'ablations'))
        from replay_real_assembly import observe_base_pose
        ob = observe_base_pose('base1')
        if ob:
            print(f"[base] observed from here: ({1000*ob[0][0]:.1f}, {1000*ob[0][1]:.1f}, {1000*ob[0][2]:.1f}) mm,"
                  f" yaw {R.from_quat(ob[1]).as_euler('xyz', degrees=True)[2]:.2f} deg")
            for o in parts:
                t = seat_grasp(o['object_name'], o['grasp_id'], ob)
                print(f"  {o['object_name']}: seat grasp flange {np.round(1000*t, 1).tolist()} mm")
        try:                                   # overlay the spots on a colour frame for a visual check
            import cv2, rclpy
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
            rclpy.init(); nn = rclpy.create_node('dis_overlay'); im = {}
            nn.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: im.__setitem__('c', m), 2)
            end = time.time() + 4
            while 'c' not in im and time.time() < end: rclpy.spin_once(nn, timeout_sec=0.05)
            img = CvBridge().imgmsg_to_cv2(im['c'], 'bgr8'); nn.destroy_node(); rclpy.shutdown()
            Tc, K, tz = depth_free_spots.last; Ti = np.linalg.inv(Tc)
            for idx, (sx, sy) in enumerate(spots):
                h = SPOT_SIZE / 2
                pts = []
                for dx, dy in [(-h, -h), (h, -h), (h, h), (-h, h)]:
                    pc = Ti[:3, :3] @ np.array([sx + dx, sy + dy, tz]) + Ti[:3, 3]
                    pts.append([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])
                cv2.polylines(img, [np.int32(pts)], True, (0, 255, 0), 3)
                cv2.putText(img, parts[idx]['object_name'] if idx < len(parts) else str(idx), tuple(np.int32(pts[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite('/tmp/disassemble_spots.png', img); print("overlay: /tmp/disassemble_spots.png")
        except Exception as e:
            print("overlay failed:", e)
        for o, s in zip(parts, spots):
            print(f"  {o['object_name']}: grasp {o['grasp_id']} width {o['gripper_width_mm']} -> spot {np.round(s, 3).tolist()}")
        return 0
    for o, (sx, sy) in zip(parts, spots):
        name, gid, w = o['object_name'], o['grasp_id'], o['gripper_width_mm']
        print(f"\n=== {name} -> ({1000*sx:.0f}, {1000*sy:.0f}) mm")
        tcp = seat_grasp(name, gid, ob)
        print(f"    seat grasp flange ({1000*tcp[0]:.1f}, {1000*tcp[1]:.1f}, {1000*tcp[2]:.1f}) mm", flush=True)
        q = [str(v) for v in SEAT_EE_QUAT]
        steps = [
            (py('primitives.move_cartesian', '--mode', 'real', '--xyz', f"{tcp[0]:.4f}", f"{tcp[1]:.4f}",
                f"{tcp[2] + HOVER_DZ:.4f}", '--quat', *q), 'hover above the seated grasp', 60),
            (py('primitives.control_gripper', f"{w:g}", '--mode', 'real'), 'open to release width', 30),
            (py('primitives.move_cartesian', '--mode', 'real', '--xyz', f"{tcp[0]:.4f}", f"{tcp[1]:.4f}",
                f"{tcp[2]:.4f}", '--quat', *q, '--min-z', '0.15', '--max-force', '12', '--min-duration', '6'),
             'descend onto the seated grasp', 60),
            (py('primitives.control_gripper', 'close', '--mode', 'real'), 'close', 30),
            (py('primitives.move_to_safe_height', '--mode', 'real'), 'pull out to safe height', 90),
        ]
        for argv, label, t in steps:
            ok, _ = run(argv, label, t)
            if not ok: return 1
        # carry at the current height, orientation kept, to above the spot
        import rclpy
        from geometry_msgs.msg import PoseStamped
        rclpy.init(); nn = rclpy.create_node('dis_tcp'); tp = {}
        nn.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: tp.__setitem__('p', m), 10)
        end = time.time() + 3
        while 'p' not in tp and time.time() < end: rclpy.spin_once(nn, timeout_sec=0.05)
        z = tp['p'].pose.position.z; nn.destroy_node(); rclpy.shutdown()
        steps = [
            (py('primitives.move_cartesian', '--mode', 'real', '--xyz', f"{sx:.4f}", f"{sy:.4f}", f"{z:.4f}",
                '--keep-orientation'), 'carry to spot', 90),
            (py('primitives.core.move_down', '--mode', 'real'), 'lower to table (contact)', 120),
            (py('primitives.control_gripper', f"{w + 6:g}", '--mode', 'real'), 'release', 30),
            (py('primitives.move_to_safe_height', '--mode', 'real'), 'lift clear', 90),
        ]
        for argv, label, t in steps:
            ok, _ = run(argv, label, t)
            if not ok: return 1
    ok, _ = run(py('primitives.move_home', '--mode', 'real'), 'home', 60)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
