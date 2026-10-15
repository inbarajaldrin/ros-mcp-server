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
TABLE_TOL = 0.006                # "flat" = within this of the table plane (paper board counts)
FINGER_CLEAR = 0.04              # clear gap between part outlines: room for the opened fingers
ANN = "/home/aldrin/Documents/aruco-grasp-annotator/data/aruco/{}_aruco.json"


def footprint_half_extents(name, base_name='base1'):
    """World xy half-extents (ex, ey) of the part as disassembly sets it down: its largest face
    (so a u part that tips over still fits), long side along the seated long axis (it is
    carried and lowered without reorienting). A fixed 100 mm square put line_green (140 mm)
    against u_brown and the fingers reaching for u_brown struck it: protective stop 2026-10-14."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    dim = json.load(open(ANN.format(name)))['cad_object_info']['dimensions']
    h = np.array([dim['length'], dim['width'], dim['height']]) / 2
    ext = np.abs(R.from_quat(load_seat_pose_in_base(base_name, name)[1]).as_matrix()) @ h
    a, b = sorted(h, reverse=True)[:2]
    return (a, b) if ext[0] >= ext[1] else (b, a)


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
    from replay_real_assembly import robot_safety_status
    st = robot_safety_status()
    if st is not None and st not in ('NORMAL', 'REDUCED'):            # never command a stopped robot
        ok, res = False, dict(res, result='failure', error=f'robot {st}')
    print(f"    {'ok' if ok else 'FAILED'} {res.get('error', '') if not ok else ''}", flush=True)
    for l in out.splitlines():                        # how deep a grasp went (pad centre vs grasp point)
        if any(k in l for k in ('Pad-centre grasp', 'Raising gripper center', 'Target gripper center')):
            print('     ' + l.split(']: ')[-1], flush=True)
    if not ok:
        print('\n'.join(out.splitlines()[-15:]))
    return ok, res


def py(*a):
    return [sys.executable, '-u', '-m', *a]


def gripper_holding(timeout=4.0):
    """True/False from the RG2's own grip-detected flag (/gripper_status 'Grasp:True'), None if
    no status arrived."""
    import rclpy
    from std_msgs.msg import String
    rclpy.init(); n = rclpy.create_node('dis_grip'); st = {}
    n.create_subscription(String, '/gripper_status', lambda m: st.__setitem__('s', m.data), 5)
    end = time.time() + timeout
    while 's' not in st and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node(); rclpy.shutdown()
    return None if 's' not in st else ('Grasp:True' in st['s'])


def depth_free_spots(extents, exclude_xy):
    """Free table spots, one per part (half-extents ex, ey), from one aligned-depth frame at
    home. A spot is a clear rectangle of the part plus FINGER_CLEAR/2 on every side; each chosen
    spot is then marked occupied with the same margin, so outlines end FINGER_CLEAR apart."""
    import rclpy
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped
    from cv_bridge import CvBridge
    # Median of N_DEPTH frames taken after a short settle. One frame grabbed right after the
    # arm reached home and the gripper opened in view ruled out most of the table (28 candidate
    # cells vs 170 two minutes later, same scene, 2026-10-14).
    time.sleep(1.0)
    rclpy.init(); n = rclpy.create_node('disassemble_depth'); st = {'d': []}
    t_from = time.time()
    n.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw',
                          lambda m: st['d'].append(m) if time.time() > t_from else None, 20)
    n.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 2)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('t', m), 10)
    end = time.time() + 8
    while time.time() < end and not (len(st['d']) >= N_DEPTH and 'i' in st and 't' in st):
        rclpy.spin_once(n, timeout_sec=0.05)
    if len(st['d']) < 3 or 'i' not in st or 't' not in st:
        n.destroy_node(); rclpy.shutdown(); return None, "no depth/camera_info/tcp"
    br = CvBridge()
    D = np.stack([br.imgmsg_to_cv2(m, 'passthrough').astype(float) for m in st['d'][-N_DEPTH:]])
    D[D == 0] = np.nan                                         # no-return pixels don't vote
    with np.errstate(all='ignore'):
        d = np.nan_to_num(np.nanmedian(D, 0), nan=0.0) / 1000.0
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
    def boxsum(m, kx, ky):                                     # sum over every kx x ky window
        S = np.pad(m.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return S[kx:, ky:] - S[:-kx, ky:] - S[kx:, :-ky] + S[:-kx, :-ky]
    m_ = FINGER_CLEAR / 2
    W, H = st['i'].width, st['i'].height
    def in_view(cx, cy, ex, ey):
        # The whole outline (plus its margin) must project inside the home image, VIEW_MARGIN_PX
        # off every edge. A spot running off the top edge left u_brown's marker at the frame
        # edge; the home pose put its grasp point 70 mm off and the descent landed on the part
        # (protective stop, 2026-10-14).
        for sx in (-1, 1):
            for sy in (-1, 1):
                pc = Tc[:3, :3].T @ (np.array([cx + sx * (ex + m_), cy + sy * (ey + m_), table_z]) - Tc[:3, 3])
                u, v = (K @ (pc / pc[2]))[:2]
                if not (VIEW_MARGIN_PX <= u <= W - VIEW_MARGIN_PX and VIEW_MARGIN_PX <= v <= H - FINGER_BAND_PX):
                    return False
        return True
    r_ = int(np.ceil(m_ / cell))                            # parts already lying there get the
    hp = np.pad(high, r_)                                      # same margin as placed ones, so every
    high = np.zeros_like(high)                                 # outline-to-outline gap is FINGER_CLEAR
    for di in range(2 * r_ + 1):
        for dj in range(2 * r_ + 1):
            high |= hp[di:di + nx, dj:dj + ny]
    spots, n_cand = [], 0
    for ex, ey in extents:
        kx, ky = int(2 * (ex + m_) / cell) + 1, int(2 * (ey + m_) / cell) + 1
        ok = (boxsum(seen, kx, ky) > 0.9 * kx * ky) & (boxsum(high, kx, ky) == 0)
        cands = []
        for i, j in zip(*np.nonzero(ok)):
            cx, cy = x0 + (i + kx / 2) * cell, y0 + (j + ky / 2) * cell
            if not (0.30 < np.hypot(cx, cy) < 0.68):
                continue
            if any(np.hypot(cx - exx, cy - eyy) < er + max(ex, ey) for exx, eyy, er in exclude_xy):
                continue
            if not in_view(cx, cy, ex, ey):
                continue
            cands.append((cx, cy))
        n_cand += len(cands)
        if not cands:
            break
        c = min(cands, key=lambda c: np.hypot(c[0] - 0.05, c[1] + 0.40))   # nearest the home view centre
        spots.append((c[0], c[1], ex, ey))
        i0, i1 = int((c[0] - ex - m_ - x0) / cell), int((c[0] + ex + m_ - x0) / cell) + 1
        j0, j1 = int((c[1] - ey - m_ - y0) / cell), int((c[1] + ey + m_ - y0) / cell) + 1
        high[max(i0, 0):i1, max(j0, 0):j1] = True               # occupied, with its margin
    spots = [(x, y) for x, y, _, _ in spots]
    depth_free_spots.last = (Tc, K, table_z)
    return spots, f"table z {1000*table_z:.1f} mm, {n_cand} candidate cells"


SEAT_EE_QUAT = (0.0, 1.0, 0.0, 0.0)        # top-down, yaw 180: every logged insert (2026-10-07/14)
HOVER_DZ = 0.10
N_DEPTH = 10                               # depth frames per free-spot map (median)
VIEW_MARGIN_PX = 60                        # free spots stay this far inside the home image
FINGER_BAND_PX = 160                       # ...and above this bottom band: move_to_grasp opens to the
                                           # pre-grasp width first and the fingertips hide it (yellow
                                           # at the bottom edge went undetected, 2026-10-15)


GRASP_OFFSETS = REPO / 'ablations/logs/grasp_tool_offsets.json'


def record_grasp_offset(name, offset_m):
    """Remember how deep `name` was last grasped (flange -> grasp point, move_to_grasp's
    grasp_tool_offset_m). The last grasp before an insert is the insert grasp."""
    try:
        d = json.load(open(GRASP_OFFSETS))
    except Exception:
        d = {}
    d[name] = dict(grasp_tool_offset_m=float(offset_m), t=time.strftime('%Y-%m-%d %H:%M:%S'))
    GRASP_OFFSETS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(d, open(GRASP_OFFSETS, 'w'), indent=1)


def grasp_offset_for(name):
    """Flange -> grasp point the insert held `name` with: recorded, else the pad-centre model."""
    try:
        return float(json.load(open(GRASP_OFFSETS))[name]['grasp_tool_offset_m'])
    except Exception:
        from primitives.shared.config import grip_width_mm, grasp_tool_offset_target_m
        return grasp_tool_offset_target_m(name, grip_width_mm(name))


def seat_grasp(name, gid, observed_base, base_name='base1'):
    """Flange pose the insert holds `name` at when seated (same chain as translate_object,
    same grasp depth as the insert: grasp_offset_for)."""
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base, predict_tcp_at_seat
    from primitives.shared.config import get_object_base_offset_m
    (bx, by, bz), bq = observed_base
    dz = float(get_object_base_offset_m(name)[2])
    _, seat_q = load_seat_pose_in_base(base_name, name)
    held = (R.from_quat(bq) * R.from_quat(seat_q)).as_quat()       # seated orientation in world
    out = predict_tcp_at_seat(base_name=base_name, object_name=name, grasp_id=gid,
                              base_world_xyz=[bx, by, bz + dz], base_world_quat_xyzw=bq,
                              held_quat_xyzw=held, ee_orientation_xyzw=SEAT_EE_QUAT,
                              flange_offset_m=grasp_offset_for(name))
    return np.array(out['predicted_tcp_at_seat']['xyz_m'])


def observe_base(base_name='base1'):
    """Read the base from straight above (as the replay's startup), then stay there: x, y and
    yaw from the depth outline of its top face (base_depth.py), the marker read as fallback."""
    from primitives.shared.config import DEFAULT_BASE_POSITION as DBP
    sys.path.insert(0, str(REPO / 'ablations'))
    from replay_real_assembly import observe_base_pose
    from base_depth import VIEW_FLANGE_Z, measure
    ok, _ = run(py('primitives.move_cartesian', '--mode', 'real', '--xyz', f"{DBP[0]:.4f}", f"{DBP[1]:.4f}",
                   f"{VIEW_FLANGE_Z:.2f}", '--quat', *map(str, SEAT_EE_QUAT)), 'base view', 60)
    if not ok:
        return None
    ob = observe_base_pose(base_name)
    if ob is None:
        return None
    time.sleep(1.0)                                            # settle before the depth frames
    ob_d, info = measure(ob, base_name)                        # x, y, yaw from the depth outline
    print(f"[base] depth outline: {json.dumps(info)}", flush=True)
    return ob_d if ob_d is not None else ob


def place_at(sx, sy, w, home=True):
    """Carry the held part at the current height (orientation kept) above (sx, sy), lower it
    onto the table (glitch-proof contact), release, lift; optionally go home."""
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
    ] + ([(py('primitives.move_home', '--mode', 'real'), 'home', 60)] if home else [])
    for argv, label, t in steps:
        ok, _ = run(argv, label, t)
        if not ok: return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--assembly-json', required=True)
    ap.add_argument('--dry-run', action='store_true', help='plan spots and print steps; no motion')
    ap.add_argument('--only', default=None)
    ap.add_argument('--relocate', default=None, metavar='PART',
                    help='pick PART off the table (move_to_grasp) and set it down at a free, '
                         'properly spaced spot; for parts lying too close to each other')
    ap.add_argument('--grasp-id', default=None,
                    help="with --relocate: grasp point to pick with, or 'auto' — the first one (centre "
                         "first) whose open fingers clear base1 and the other parts (spacing_check)")
    ap.add_argument('--place-held', default=None, metavar='PART',
                    help='the gripper holds PART (e.g. a run stopped mid-insert): set it down in the '
                         'staging region and stay above it; then replay --only PART --skip-startup '
                         '--part-at-staging regrasps and inserts it')
    ap.add_argument('--pick-view-dy', type=float, default=0.0,
                    help='with --relocate: shift the arm this far in world y from home (top-down, same '
                         'height) before the pick, for a part home sees badly (fingers hide the bottom '
                         'of the image); the spot is still planned from home')
    ap.add_argument('--from-staging', action='store_true',
                    help='with --relocate: the part sits in the staging region (not visible from '
                         'home) — plan the spot from home, then pick it over the clear area')
    ap.add_argument('--to-staging', action='store_true',
                    help='with --only: take the seated part out with the insert grasp and set it down '
                         'in the staging region (clear area) for a fresh regrasp, e.g. a tilted seat')
    a = ap.parse_args()
    if os.nice(0) < 5:                     # the UR driver has no RT priority here: run below it
        os.nice(5 - os.nice(0))
    if a.to_staging and not a.only:
        ap.error('--to-staging needs --only PART')
    if a.place_held:
        from primitives.shared.config import GRIP_WIDTH_MM
        if a.place_held not in GRIP_WIDTH_MM:
            ap.error(f'unknown part {a.place_held}')
        if gripper_holding() is False:
            print(f"the gripper reports nothing held — not placing {a.place_held}"); return 1
        return place_at(STAGING_XY[0], STAGING_XY[1], GRIP_WIDTH_MM[a.place_held], home=False)
    order = json.load(open(REPO / 'ablations/eval_resources/fmb1_assembly.json'))['assembly_order']
    sel = a.relocate or a.only
    parts = [o for o in sorted(order, key=lambda o: -o['position']) if not sel or o['object_name'] == sel]
    from primitives.shared.config import DEFAULT_BASE_POSITION
    if not a.dry_run:
        ok, _ = run(py('primitives.move_home', '--mode', 'real'), 'home', 60)
        if not ok: return 1
        ok, _ = run(py('primitives.control_gripper', 'open', '--mode', 'real'), 'open', 30)
        if not ok: return 1
    # base1 is 35 mm tall: the depth map marks it occupied and the FINGER_CLEAR/2 dilation keeps
    # every spot 20 mm off its edges. A circle around its half-diagonal (190 mm with the part) also
    # swallowed the real space beside its square sides (2026-10-14), so only the staging region is
    # excluded explicitly (a part may be set down there unseen from home).
    exclude = [(*STAGING_XY, 0.06 + FINGER_CLEAR)]
    extents = [footprint_half_extents(o['object_name']) for o in parts]
    print("[spots] footprints mm: " + ", ".join(f"{o['object_name']} {2000*e[0]:.0f}x{2000*e[1]:.0f}"
                                                for o, e in zip(parts, extents)))
    if a.to_staging:
        spots, note = [tuple(STAGING_XY)] * len(parts), "staging region (--to-staging)"
    else:
        spots, note = depth_free_spots(extents, exclude)
    print(f"[spots] {note}: {[(round(1000*x), round(1000*y)) for x, y in (spots or [])]} mm")
    if not spots or len(spots) < len(parts):
        print("not enough free spots seen from here — stopping")
        if not a.dry_run: return 1
    if a.relocate and not a.dry_run:
        o = parts[0]; (sx, sy) = spots[0]
        name, gid, w = o['object_name'], o['grasp_id'], o['gripper_width_mm']
        if a.grasp_id == 'auto':
            # A part lying against base1 or a neighbour: grasp where both open fingers have room
            # (2026-10-15: line_green 7 mm from base1 — its centre grasp put a finger on the base).
            from spacing_check import clear_grasp, observe_parts, PARTS
            from replay_real_assembly import observe_base_pose
            seen = observe_parts(PARTS)
            ob_ = observe_base_pose('base1')
            if name not in seen or ob_ is None:
                print(f"{name} or base1 not seen — stopping"); return 1
            gid, rep = clear_grasp(name, seen[name], ob_, seen, w)
            print(f"[grasp] {name}: finger clearance per grasp id (mm) {rep} -> {gid}", flush=True)
            if gid is None:
                print(f"no grasp point of {name} leaves the open fingers room — stopping"); return 1
        elif a.grasp_id is not None:
            gid = int(a.grasp_id)
        print(f"\n=== relocate {name} (grasp {gid}) -> ({1000*sx:.0f}, {1000*sy:.0f}) mm")
        pick = [(py('primitives.core.move_to_clear_area', '--mode', 'real'), 'over staging', 90)] if a.from_staging else []
        if a.pick_view_dy:
            from primitives.shared.config import HOME_POSE as _H
            pick.append((py('primitives.move_cartesian', '--mode', 'real', '--xyz', f"{_H[0]:.4f}",
                            f"{_H[1] + a.pick_view_dy:.4f}", f"{_H[2]:.4f}", '--keep-orientation'),
                         'shift the view', 60))
        for argv, label, t in pick + [
            (py('primitives.move_to_grasp', '--object-name', name, '--grasp-id', str(gid), '--mode', 'real',
                '--pre-grasp-width', f"{w:g}", '--pad-center'), 'move_to_grasp', 180),
            (py('primitives.control_gripper', 'close', '--mode', 'real'), 'close', 30),
            (py('primitives.move_to_safe_height', '--mode', 'real'), 'lift', 90)]:
            ok, _ = run(argv, label, t)
            if not ok: return 1
        return place_at(sx, sy, w)
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
                hx, hy = extents[idx][0] + FINGER_CLEAR / 2, extents[idx][1] + FINGER_CLEAR / 2
                pts = []
                for dx, dy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
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
    for i, (o, (sx, sy)) in enumerate(zip(parts, spots)):
        name, gid, w = o['object_name'], o['grasp_id'], o['gripper_width_mm']
        print(f"\n=== {name} -> ({1000*sx:.0f}, {1000*sy:.0f}) mm")
        if i > 0:
            # The arm rides a 7th-axis slider: each pull-out can creep the carriage, so the base
            # moves in the robot's frame. Read it again before every seat grasp (see
            # replay_real_assembly.reread_base).
            prev, ob = ob, observe_base()
            if ob is None:
                print("base1 not observed — stopping"); return 1
            print(f"[base] moved ({1000*(ob[0][0]-prev[0][0]):+.1f}, {1000*(ob[0][1]-prev[0][1]):+.1f}) mm "
                  f"since the last read", flush=True)
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
            ok, res = run(argv, label, t)
            if not ok: return 1
            # The grip must hold through the pull-out: 2026-10-15 line_green came out of the jaws
            # on the way up, fell beside base1, and every later step reported ok (the gripper
            # carried nothing to its spot); the next pick then put an open finger on the base.
            if label in ('close', 'pull out to safe height'):
                width = res.get('final_width_mm')
                from primitives.shared.config import GRIP_WIDTH_MM
                want = GRIP_WIDTH_MM.get(name)
                held = gripper_holding()
                if held is False or (label == 'close' and width is not None and want is not None
                                     and width < want - 6.0):
                    print(f"\n!!! {name}: not held after '{label}' (grip flag {held}, width {width} mm, "
                          f"part {want} mm) — stopping", flush=True)
                    return 1
        from replay_real_assembly import forget_seat_offset
        forget_seat_offset(name)                            # out of the base: its seat offset is gone
        if place_at(sx, sy, w, home=False) != 0:
            return 1
    if a.to_staging:                                       # stay over the clear area: the regrasp sees it
        return 0
    ok, _ = run(py('primitives.move_home', '--mode', 'real'), 'home', 60)
    if not ok:
        return 1
    # Where did the parts actually land? The spots were planned clear, but a part that slips or
    # tips lands elsewhere (2026-10-15: line_green 7 mm from base1). Check from home, fail closed.
    from spacing_check import check as spacing, observe_parts, PARTS
    seen = observe_parts(PARTS)
    ok_sp, rep = spacing(seen, ob)
    rep['unseen'] = [n for n in PARTS if n not in seen]
    print(f"[spacing] {json.dumps(rep)}", flush=True)
    if not ok_sp:
        print("\n!!! parts too close for the open gripper after disassembly — stopping")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
