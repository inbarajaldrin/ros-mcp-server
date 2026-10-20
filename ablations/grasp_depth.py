#!/usr/bin/env python3
"""Per-part grasp depth: pad centre on the grasp point, unless the fingers would hit something
when the part is seated (2026-10-14).

    python3 -u ablations/grasp_depth.py [--assembly fmb1|fmb2|fmb3] [--write]

The RG2 pad centre belongs on the grasp point (config.pad_center_offset_m). But a grasp cannot
change between pick and insert, and at the seat the fingertips must stay above whatever lies
under them: the base's top, or a part seated earlier. For each part, in assembly order, this puts
the finger tip blocks (RG2 CAD: 11.4 mm thick along the closing axis, 20.2 mm wide) beside the
grasp point at the seat, finds the highest CAD surface of the base + the parts already seated under
them, and limits the grasp depth so the tips stay SEAT_CLEARANCE_M above it.
Updates that assembly's parts in ablations/eval_resources/grasp_depth.json (read by
config.grasp_tool_offset_floor_m; was fmb1_grasp_depth.json until 2026-10-20).
Meshes: aruco-grasp-annotator OBJ (cm, object frame). Seat poses: cad_lookup.load_seat_pose_in_base.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = Path('/home/aldrin/Documents/aruco-grasp-annotator/data')
OUT = REPO / 'ablations/eval_resources/grasp_depth.json'
SEAT_CLEARANCE_M = 0.004        # base1 z is snapped to the config table, not measured: keep 4 mm
TIP_THICK_M = 0.0114 + 0.002        # RG2 tip block along the closing axis (CAD) + margin
TIP_WIDTH_M = 0.0202 + 0.004        # across (CAD) + margin


def load_mesh(name):
    v, f = [], []
    for l in open(DATA / f'models/{name}.obj'):
        if l.startswith('v '):
            v.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('f '):
            idx = [int(t.split('/')[0]) - 1 for t in l.split()[1:]]
            f += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    return np.array(v) * 0.01, np.array(f)


def surface_points(V, F, step=0.0015):
    """Dense samples on every triangle (the meshes are coarse prisms)."""
    pts = [V]
    for a, b, c in F:
        A, B, C = V[a], V[b], V[c]
        n = int(max(np.linalg.norm(B - A), np.linalg.norm(C - A), np.linalg.norm(C - B)) / step) + 1
        u, w = np.meshgrid(np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1))
        m = u + w <= 1
        pts.append(A + np.outer(u[m], B - A) + np.outer(w[m], C - A))
    return np.vstack(pts)


def pose(name, base='base1'):
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    p, q = load_seat_pose_in_base(base, name)
    T = np.eye(4); T[:3, :3] = R.from_quat(q).as_matrix(); T[:3, 3] = p
    return T


def grasp(name, gid):
    d = json.load(open(DATA / f'grasp_points/{name}_grasp_points.json'))
    g = next(g for g in d['grasp_points'] if g['id'] == gid)
    p = np.array([g['position'][k] for k in 'xyz'])
    av = np.array([g['approach_vector'][k] for k in 'xyz'])
    axis = np.array([1.0, 0, 0]) if g['grasp_validity'].get('x_axis_gripper_width_mm') else np.array([0, 1.0, 0])
    return p, av, axis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--assembly', default='fmb1', help='fmb1 | fmb2 | fmb3 (eval_resources/<name>_assembly.json)')
    a = ap.parse_args()
    from primitives.shared.config import GRIP_WIDTH_MM, rg2_tip_length_m, pad_center_offset_m
    asm = json.load(open(REPO / f'ablations/eval_resources/{a.assembly}_assembly.json'))
    base = asm['base_name']
    order = sorted(asm['assembly_order'], key=lambda o: o['position'])
    world = {}                                                   # name -> surface points, base frame
    V, F = load_mesh(base); world[base] = surface_points(V, F)
    out = json.load(open(OUT)) if OUT.exists() else {}
    out.update({'_doc': __doc__.strip().splitlines()[0], 'seat_clearance_m': SEAT_CLEARANCE_M})
    out.setdefault('parts', {})
    for o in order:
        name, gid = o['object_name'], o['grasp_id']
        T = pose(name, base)
        p, av, axis = grasp(name, gid)
        gp = T[:3, :3] @ p + T[:3, 3]                           # grasp point at the seat (base frame)
        up = np.array([0.0, 0.0, 1.0])     # every insert is top-down (SEAT_EE_QUAT), whatever the part's own
                                            # approach vector (u parts stand in the seat but lie on the table)
        cl = T[:3, :3] @ axis; cl -= up * (cl @ up); cl /= np.linalg.norm(cl)
        across = np.cross(up, cl)
        gap = GRIP_WIDTH_MM[name] / 1000.0
        surf = np.vstack(list(world.values()))
        rel = surf - gp
        s_cl, s_ac, s_up = rel @ cl, rel @ across, rel @ up
        under = (np.abs(s_ac) < TIP_WIDTH_M / 2) & (np.abs(s_cl) > gap / 2) & (np.abs(s_cl) < gap / 2 + TIP_THICK_M) \
                & (s_up < 0)
        top = float(s_up[under].max()) if under.any() else -1.0  # highest surface under a finger, rel. grasp point
        whom = [k for k, P in world.items() if (lambda r: ((np.abs(r @ across) < TIP_WIDTH_M / 2) & (np.abs(r @ cl) > gap / 2)
                & (np.abs(r @ cl) < gap / 2 + TIP_THICK_M) & (r @ up < 0) & (r @ up > top - 1e-4)).any())(P - gp)]
        tip, pad = rg2_tip_length_m(GRIP_WIDTH_MM[name]), pad_center_offset_m(GRIP_WIDTH_MM[name])
        # tips sit (tip - L) below the grasp point; keep them SEAT_CLEARANCE_M above `top`
        floor = tip - (-top - SEAT_CLEARANCE_M)
        L = max(pad, floor)
        out['parts'][name] = dict(base=base, grasp_id=gid, gap_mm=GRIP_WIDTH_MM[name], pad_center_offset_m=round(pad, 5),
                                  seat_floor_offset_m=round(floor, 5), grasp_tool_offset_m=round(L, 5),
                                  limited_by=(whom[0] if L > pad else 'none'),
                                  surface_below_grasp_point_mm=round(-1000 * top, 1),
                                  pad_centre_above_grasp_point_mm=round(1000 * (pad - L) * -1, 1) if L > pad else 0.0)
        print(f"{name:18s} gap {GRIP_WIDTH_MM[name]:4.1f} mm | surface under fingers {-1000*top:5.1f} mm below the grasp point"
              f" ({', '.join(whom) or 'nothing'}) | pad centre {1000*pad:.1f}, seat floor {1000*floor:.1f}"
              f" -> flange->grasp point {1000*L:.1f} mm (legacy 228.6; {1000*(0.2286-L):+.1f} mm deeper)"
              + (f", pad centre {1000*(L-pad):.1f} mm above the grasp point" if L > pad else ", pad centred"))
        V, F = load_mesh(name); P = surface_points(V, F)
        world[name] = P @ T[:3, :3].T + T[:3, 3]
    if a.write:
        json.dump(out, open(OUT, 'w'), indent=1)
        print(f"wrote {OUT}")


if __name__ == '__main__':
    main()
