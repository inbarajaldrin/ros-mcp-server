#!/usr/bin/env python3
"""Generated start layouts for the disassembly: parts laid flat in the staging region (2026-10-19).

    python3 -u ablations/staging_layout.py [--cycle N] [--all-flat] [--seed S] [--look]

The IDEA of the operator's sim task `fmb1_assemble_ur5e_rg2_flat_gen` (dual-a4500,
plan-anything/data/tasks/.../task.yaml), not its numbers — the sim mounts the robot at table height,
the cell does not (base 82 mm above the table):

  * Body2/4/5 (u_brown, u_orange, inverted_u_yellow) lie FLAT on a large face, yaw free, flip free
    (either large face up); Body3 (line_green) lies along its length, yaw free;
  * every part wholly inside the staging rectangle, no overlap, >= 25 mm between parts.

The sim's staging rectangle (table frame x [-0.51, -0.19], y [-0.2785, 0.0415], table at world y
0.4985; world = ROS base_link) is, in the UR base frame (base_link turned 180 deg about z),
x [0.19, 0.51], y [-0.54, -0.22]; the sim's "reorient" rectangle lands on our clear area
(-0.32, -0.50), which confirms the mapping. The whole rectangle is clear (depth, 2026-10-19),
reachable top-down at pick height (IK from home), and seen from STAGING_VIEW, not from home
(home sees x <= ~0.42) — the assembly looks from there for parts it cannot see from home.

Cell differences from the sim: a part reaches flat by being set down upright (as seated) at the drawn
yaw and TIPPED (tip_part.py: closed fingers strike its top at an angle, <= 5 N). By default one
tippable part per cycle stays upright (rotating: u_brown, u_orange, inverted_u_yellow), so the
assembly runs both the 1-pass and the 2-pass path; --all-flat tips all three. The wrist turns at most
90 deg from the seated yaw to set a part down (the D415 cable/plug is worn), so for a tipped part the
opening direction and the face down are not independent; both are covered over the draws.
"""
import argparse
import functools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))

REGION = {'x': (0.19, 0.51), 'y': (-0.54, -0.22)}   # UR base frame (see docstring)
STAGING_VIEW = (0.35, -0.33, 0.62)                   # flange, top-down: the whole region in view AND above
                                                     # the image's bottom 160 px, where the fingers hide the
                                                     # table (from (0.35,-0.38,0.56) line_green at y -0.29 went
                                                     # unseen, 2026-10-19): region rows 55..536 of 720
TOP_DOWN_Q = (0.0, 1.0, 0.0, 0.0)
CLEARANCE_M = 0.025                                  # generator's clearance between parts
SWEEP_CLEARANCE_M = 0.010                            # finger start zone: its lowering is guarded (stops on touch)
LAND_OVERSHOOT_M = 0.020                             # a tipped part slides on: yellow landed 13 mm past plan
LAND_SIDE_M = 0.010                                  # ... and a little sideways (2026-10-19, 10 mm from u_brown)
TIPPABLE = ('u_brown', 'u_orange', 'inverted_u_yellow')
STRIKE_BACK_M = 0.045                                # behind the near face: finger start zone
WRIST_TURN_MAX = math.radians(90)                    # from the seated yaw (worn camera cable)
RAISED_M = 0.010                                     # depth: higher than this above the table = occupied
OUT = REPO / 'ablations/logs/staging_layout.json'

# Standing (seated stance) size: length L along the part's x, thickness T along y, height H (m).
# Strike offset along x: U parts stand open-top (legs x in +-[14, 35] mm, 28 mm gap) -> a leg;
# yellow has a solid 42 mm block on top -> its centre (CAD profile, 2026-10-19).
STAND = {'u_brown': dict(L=0.070, T=0.021, H=0.070, strike_x=0.0245),
         'u_orange': dict(L=0.070, T=0.021, H=0.070, strike_x=0.0245),
         'inverted_u_yellow': dict(L=0.1127, T=0.021, H=0.070, strike_x=0.0),
         'line_green': dict(L=0.0259, T=0.140, H=0.0259, strike_x=0.0)}   # already flat: never tipped


GRASP_ID = {'u_brown': 1, 'u_orange': 1, 'line_green': 1, 'inverted_u_yellow': 2}   # fmb1_assembly.json


@functools.lru_cache(maxsize=None)
def grasp_offset(name):
    """(along the part's x, along its y) from the part's centre to its grasp point, standing as seated
    — the gripper holds the U parts by a LEG (+25.5 mm): place_xy is the part's centre, the flange
    goes to centre + this (2026-10-19: aiming by the flange put u_orange's strike 50 mm out, past the
    part's end). CAD: aruco-grasp-annotator grasp points + mesh bbox centre + seat pose."""
    from grasp_depth import load_mesh
    from scipy.spatial.transform import Rotation as R
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    d = json.load(open(f"/home/aldrin/Documents/aruco-grasp-annotator/data/grasp_points/{name}_grasp_points.json"))
    gp = next(g for g in d['grasp_points'] if g['id'] == GRASP_ID[name])['position']
    V, _ = load_mesh(name)
    off = R.from_quat(load_seat_pose_in_base('base1', name)[1]).apply(
        np.array([gp['x'], gp['y'], gp['z']]) - (V.min(0) + V.max(0)) / 2)
    return float(off[0]), float(off[1])               # base frame at seated yaw 0 = part x, y


def rect(c, u, a, b0, b1, v=None):
    """Corners of a rectangle centred on c along unit u (half-length a) and spanning [b0, b1]
    along v (default u turned +90 deg)."""
    v = np.array([-u[1], u[0]]) if v is None else v
    return np.array([c + s * a * u + t * v for s, t in ((-1, b0), (1, b0), (1, b1), (-1, b1))])


def _sep_axis_gap(P, Q):
    """Smallest separating distance between two convex polygons (0 if they overlap)."""
    best = 0.0
    for poly in (P, Q):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            pa, pb = P @ n, Q @ n
            best = max(best, max(pb.min() - pa.max(), pa.min() - pb.max()))
    return best


def inside_region(poly, margin=0.0):
    return (poly[:, 0].min() >= REGION['x'][0] + margin and poly[:, 0].max() <= REGION['x'][1] - margin
            and poly[:, 1].min() >= REGION['y'][0] + margin and poly[:, 1].max() <= REGION['y'][1] - margin)


def part_plan(name, flat, place_xy, yaw, sign):
    """Geometry of one part: where it is set down upright, which way it is tipped (sign * n), the
    final outline on the table, and the area the tip sweeps (finger start zone .. landing edge)."""
    g = STAND[name]
    u = np.array([math.cos(yaw), math.sin(yaw)])           # part x (length) in the world
    n = np.array([-u[1], u[0]])                            # part y (thickness)
    c = np.asarray(place_xy, float)
    gx, gy = grasp_offset(name)
    flange = (c + gx * u + gy * n).tolist()               # where the gripper sets it down
    if name == 'line_green':                               # lies flat as seated: L along x is 25.9
        final = rect(c, n, g['T'] / 2, -g['L'] / 2, g['L'] / 2, v=u)
        return dict(name=name, flat=True, tip=False, place_xy=c.tolist(), flange_xy=flange, yaw=yaw,
                    final=final, sweep=final)
    if not flat:
        final = rect(c, u, g['L'] / 2, -g['T'] / 2, g['T'] / 2, v=n)
        return dict(name=name, flat=False, tip=False, place_xy=c.tolist(), flange_xy=flange, yaw=yaw,
                    final=final, sweep=final)
    d = sign * n                                           # tip direction: the top falls this way
    # The landing is not exact: reserve room for the part sliding on and turning a little.
    final = rect(c, u, g['L'] / 2 + LAND_SIDE_M, -g['T'] / 2, g['H'] - g['T'] / 2 + LAND_OVERSHOOT_M, v=d)
    sweep = rect(c, u, g['L'] / 2 + LAND_SIDE_M, -(g['T'] / 2 + STRIKE_BACK_M), g['H'] - g['T'] / 2 + LAND_OVERSHOOT_M, v=d)
    land = c + d * (g['H'] / 2 - g['T'] / 2)
    return dict(name=name, flat=True, tip=True, place_xy=c.tolist(), flange_xy=flange, yaw=yaw, tip_dir=d.tolist(),
                land_xy=land.tolist(), strike_x=g['strike_x'], final=final, sweep=sweep)


def min_gap(L):
    """Closest pair of reserved landings in a layout (m)."""
    F = [np.asarray(p['final']) for p in L['parts']]
    return min((_sep_axis_gap(F[i], F[j]) for i in range(len(F)) for j in range(i + 1, len(F))), default=1.0)


def draw(order, cycle=0, all_flat=False, seed=None, occupied=None, base_yaw=0.0, attempts=4000, best_of=30):
    """The most spread-out of `best_of` admitted random layouts for `order` (disassembly order), or None.
    Each draw follows the generator's rules (random position, yaw, flip; >= 25 mm); the first admitted
    draw sat at a median 32 mm closest pair, and a part sliding or falling on ate that (2026-10-19) —
    the region is only ~24 % used, so keep the draw whose closest pair is farthest apart.
    occupied: (N, 2) robot-xy points raised above the table in the staging view."""
    seed = int(time.time()) if seed is None else seed
    best = None
    for k in range(best_of):
        L = _draw_one(order, cycle, all_flat, seed * 1000 + k, occupied, base_yaw, attempts)
        if L is not None and (best is None or min_gap(L) > min_gap(best)):
            best = L
    if best is not None:
        best['seed'], best['min_gap_mm'] = seed, round(1000 * min_gap(best), 1)
    return best


def _draw_one(order, cycle, all_flat, seed, occupied, base_yaw, attempts):
    rng = np.random.default_rng(seed)
    upright = None if all_flat else TIPPABLE[cycle % len(TIPPABLE)]
    occ = np.zeros((0, 2)) if occupied is None else np.asarray(occupied)
    for restart in range(attempts // 200):
        plans = []
        for name in order:
            for _ in range(200):
                turn = rng.uniform(-WRIST_TURN_MAX, WRIST_TURN_MAX)
                yaw = base_yaw + turn
                sign = 1.0 if rng.uniform() >= 0.5 else -1.0
                xy = (rng.uniform(*REGION['x']), rng.uniform(*REGION['y']))
                p = part_plan(name, name != upright, xy, yaw, sign)
                p['wrist_turn'] = turn
                if not (inside_region(p['final']) and inside_region(p['sweep'])):
                    continue
                if any(_sep_axis_gap(p['final'], q['final']) < CLEARANCE_M
                       or min(_sep_axis_gap(p['sweep'], q['final']), _sep_axis_gap(p['final'], q['sweep']))
                       < SWEEP_CLEARANCE_M for q in plans):
                    continue
                if len(occ):
                    poly = p['sweep']
                    lo, hi = poly.min(0) - CLEARANCE_M, poly.max(0) + CLEARANCE_M
                    if np.any(np.all((occ >= lo) & (occ <= hi), axis=1)):
                        continue
                plans.append(p)
                break
            else:
                break                                      # re-draw the whole arrangement
        if len(plans) == len(order):
            return dict(seed=seed, cycle=cycle, upright=upright, all_flat=all_flat,
                        parts=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
                               for p in plans])
    return None


def look(say=print):
    """Move over the staging region (flange STAGING_VIEW, top-down) and return (table_z, raised_xy)
    from the median of 10 aligned depth frames. Points over 150 mm above the table are the gripper."""
    import subprocess, os
    env = dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")
    r = subprocess.run([sys.executable, '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                        *[f"{v:.4f}" for v in STAGING_VIEW], '--quat', *map(str, TOP_DOWN_Q)],
                       capture_output=True, text=True, timeout=90, env=env, cwd=str(REPO))
    if '"result": "success"' not in r.stdout:
        say("staging view move failed"); return None, None
    time.sleep(1.5)
    return depth_raised(say)


def depth_raised(say=print):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import CameraInfo, Image
    from scipy.spatial.transform import Rotation as R
    rclpy.init(); n = rclpy.create_node('staging_look'); st = {'d': []}
    n.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', lambda m: st['d'].append(m), 5)
    n.create_subscription(CameraInfo, '/camera/camera/aligned_depth_to_color/camera_info', lambda m: st.__setitem__('i', m), 1)
    n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: st.__setitem__('t', m), 1)
    end = time.time() + 15
    while (len(st['d']) < 10 or 'i' not in st or 't' not in st) and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node(); rclpy.shutdown()
    if len(st['d']) < 10 or 'i' not in st or 't' not in st:
        say("no depth / pose for the staging look"); return None, None
    D = np.median(np.stack([np.frombuffer(m.data, np.uint16).reshape(m.height, m.width) for m in st['d'][-10:]]), 0) / 1000.0
    K = np.array(st['i'].k).reshape(3, 3); p, o = st['t'].pose.position, st['t'].pose.orientation
    ee = R.from_quat([o.x, o.y, o.z, o.w])
    from seat_check import X_CAM            # the localizer config's mount (single source, 2026-10-20)
    cam_p = np.array([p.x, p.y, p.z]) + ee.apply(X_CAM[:3, 3])
    cam_r = ee * R.from_matrix(X_CAM[:3, :3])
    h, w = D.shape; u, v = np.meshgrid(np.arange(w), np.arange(h))
    X = np.stack([(u - K[0, 2]) / K[0, 0] * D, (v - K[1, 2]) / K[1, 1] * D, D], -1)[D > 0.2]
    W = cam_r.apply(X) + cam_p
    inr = (W[:, 0] >= REGION['x'][0]) & (W[:, 0] <= REGION['x'][1]) & (W[:, 1] >= REGION['y'][0]) & (W[:, 1] <= REGION['y'][1])
    tz = float(np.median(W[inr, 2]))
    raised = W[inr & (W[:, 2] > tz + RAISED_M) & (W[:, 2] < tz + 0.15)][:, :2]
    say(f"[staging] table z {1000*tz:.1f} mm, {len(raised)} raised points in the region")
    return tz, raised


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cycle', type=int, default=0)
    ap.add_argument('--all-flat', action='store_true')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--look', action='store_true', help='move over the staging region and check depth first')
    a = ap.parse_args()
    order = [o['object_name'] for o in sorted(json.load(open(REPO / 'ablations/eval_resources/fmb1_assembly.json'))
                                              ['assembly_order'], key=lambda o: -o['position'])]
    occ = None
    if a.look:
        tz, occ = look()
        if tz is None:
            return 1
    L = draw(order, a.cycle, a.all_flat, a.seed, occ)
    if L is None:
        print("no admissible layout"); return 1
    for p in L['parts']:
        print(f"  {p['name']:18s} {'FLAT (tip)' if p.get('tip') else ('flat' if p['flat'] else 'UPRIGHT'):10s} "
              f"set down at ({1000*p['place_xy'][0]:.0f}, {1000*p['place_xy'][1]:.0f}) mm, wrist turn "
              f"{math.degrees(p['wrist_turn']):+.0f} deg" + (f", tip toward ({p['tip_dir'][0]:+.2f}, {p['tip_dir'][1]:+.2f})"
                                                              if p.get('tip') else ''))
    json.dump(L, open(OUT, 'w'), indent=1)
    print(f"seed {L['seed']}, upright: {L['upright']} -> {OUT}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
