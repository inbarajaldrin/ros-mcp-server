#!/usr/bin/env python3
"""Held part's xy error from two light side-face touches against its base (2026-10-21).

Operator 10-21: "when the gripper is closed you have the perfect chain from base to gripper ... that is the whole
point of force feedback, it accounts for any slippage; fine-tune when you touch a flat surface and what force is
enough". A table touch at 1.5 N measured u_brown's bottom 1.9 mm off the chain and its contact centroid where the
grasp puts it (level, no pivot) - z and stance are fine; what misses is xy (the pocket admits the part within about
+-1 x +-0.5 mm; camera reads spread 2-5 mm between views).

Here the part touches the base's outer -y face with its +y face, then the -x face with its +x face, each from 8 mm
outside, bottom 15 mm below the base top, horizontally at 3 mm/s, stopping on a sustained |dF| > 1.5 N
(primitives.move_cartesian --max-force). Where the chain says the face is at contact vs where the base face is =
the part's error in the hand along that axis. The pocket sits at a CAD-fixed place relative to the base faces, so an
error in the base read cancels (it moves faces and pocket together). Returns the world xy error (real - chained)
the insert subtracts from its aim.

    from compliant_insertion_studio.wrapper.face_touch import measure
    d_world_xy, info = measure('u_brown', 'base1', base_pose, held_q, grasp_id=1)
"""
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
CLEAR_M = 0.025              # start this far outside the face (8 mm: the descent beside it bumped at 3.3 N)
BELOW_TOP_M = 0.015          # part bottom this far below the base top during the touch
OVERTRAVEL_M = 0.012         # travel past the expected contact before giving up
MAX_FORCE_N = 1.5
SPEED_M_S = 0.003
MAX_ERR_M = 0.012            # a larger 'error' is a wrong touch (another surface), not the hand


def _move(xyz, quat, max_force=None, min_duration=None):
    argv = [sys.executable, '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
            *[f"{v:.5f}" for v in xyz], '--quat', *[f"{v:.6f}" for v in quat], '--min-z', '0.15']
    if max_force is not None:
        argv += ['--max-force', f"{max_force:g}"]
    if min_duration:
        argv += ['--min-duration', f"{min_duration:.2f}"]
    p = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}"))
    lines = p.stdout.splitlines()
    try:
        return json.loads(lines[lines.index('__RESULT_JSON__') + 1])
    except Exception:
        return {'result': 'failure', 'error': lines[-1] if lines else 'no output'}


def _tcp():
    from compliant_insertion_studio.wrapper.marker_yaw import _tcp_pose_now
    p, q = _tcp_pose_now()
    return np.asarray(p), np.asarray(q)


def Rb_z(bq, z_local):
    return float(z_local)                                            # base is flat on the table (roll/pitch < 1.5 deg)


def measure(obj, base, base_pose, held_q, grasp_id=1, tcp_to_bottom_m=0.2646, say=print):
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base, predict_tcp_at_seat
    from compliant_insertion_studio.wrapper.cad_geometry import _load_base_top_z_local
    from grasp_depth import load_mesh, surface_points
    from primitives.shared.config import held_tool_offset, held_tool_offset_for
    bp, bq = np.asarray(base_pose[0], float), base_pose[1]
    Rb = R.from_quat(bq)
    sp, s_q = load_seat_pose_in_base(base, obj)
    V, F = load_mesh(obj); Pb = R.from_quat(s_q).apply(surface_points(V, F, 0.002)) + sp     # part at its seat, base frame
    # Fold branch (2026-10-21, the fold-branch trap): for a symmetric part the chain's seat TCP depends on the
    # orientation fed in (u_brown: 186.7 vs 200.8 mm). Keep the branch whose TCP-to-bottom matches the measured
    # table touch (the part's real lowest point under the flange).
    bottom_seat_w = bp[2] + Rb_z(bq, Pb[:, 2].min())
    from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
    from utils.data_path_finder import get_symmetry_dir
    p0, q0 = _tcp()
    try:
        cands = [R.from_matrix(m_).as_quat() for m_ in
                 equivalent_orientations(R.from_quat(held_q).as_matrix(), load_symmetry_data(obj, str(get_symmetry_dir())))]
    except Exception:
        cands = [np.asarray(held_q)]
    best = None
    for hq in cands:
        pr = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=bp,
                                 base_world_quat_xyzw=bq, held_quat_xyzw=list(hq), ee_orientation_xyzw=list(q0),
                                 flange_offset_m=float(held_tool_offset_for(obj)[2]))
        z = pr['predicted_tcp_at_seat']['xyz_m'][2]
        miss = abs((z - bottom_seat_w) - tcp_to_bottom_m)
        if best is None or miss < best[0]:
            best = (miss, pr)
    pred = best[1]
    say(f"  [face-touch] {obj}: fold branch with TCP-to-bottom {1000*(pred['predicted_tcp_at_seat']['xyz_m'][2] - bottom_seat_w):.1f} "
        f"mm (touch-measured {1000*tcp_to_bottom_m:.1f})")
    if best[0] > 0.006:
        return None, {'error': f'no fold branch within 6 mm of the measured TCP-to-bottom ({1000*best[0]:.1f} mm)'}
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); sq = np.array(pred['predicted_tcp_at_seat']['quat_xyzw'])
    Vb, Fb = load_mesh(base); B = surface_points(Vb, Fb, 0.003)
    bx0, bx1, by0, by1 = B[:, 0].min(), B[:, 0].max(), B[:, 1].min(), B[:, 1].max()
    top = _load_base_top_z_local(base); bot = Pb[:, 2].min()
    z_shift = (top - BELOW_TOP_M) - bot                              # base-frame lift of the part from its seat

    def tcp_at(shift_b):                                             # TCP for the part shifted by shift_b (base frame)
        return st + Rb.apply(shift_b)

    out = {}
    # the faces that can touch: the part's extent within the band that goes below the base top during the touch
    # (u_brown's +x extent over the whole part is not where its bottom 15 mm ends - the x touch ran 37 mm, no contact)
    band = Pb[Pb[:, 2] <= bot + BELOW_TOP_M - 0.002]
    say(f"  [face-touch] {obj}: touching band (lowest {1000*(BELOW_TOP_M-0.002):.0f} mm): x {1000*band[:,0].min():.1f}.."
        f"{1000*band[:,0].max():.1f}, y {1000*band[:,1].min():.1f}..{1000*band[:,1].max():.1f} mm (whole part x "
        f"{1000*Pb[:,0].min():.1f}..{1000*Pb[:,0].max():.1f}, y {1000*Pb[:,1].min():.1f}..{1000*Pb[:,1].max():.1f})")
    for axis, face, part_face in (('y', by0, band[:, 1].max()), ('x', bx0, band[:, 0].max())):
        k = 1 if axis == 'y' else 0
        other = Pb[:, 1 - k].mean() if False else 0.0
        s_start = np.zeros(3); s_start[k] = (face - CLEAR_M) - part_face; s_start[2] = z_shift
        s_end = s_start.copy(); s_end[k] += CLEAR_M + OVERTRAVEL_M
        above = tcp_at(s_start) + np.array([0, 0, 0.04])
        for xyz, lab in ((above, 'over the start'), (tcp_at(s_start), 'down beside the face')):
            r = _move(xyz, sq, max_force=3.0 if lab.startswith('down') else None,
                      min_duration=4.0 if lab.startswith('down') else None)
            if r.get('result') == 'failure':
                pz, _ = _tcp()
                say(f"  [face-touch] {obj} {axis}: {lab} failed ({r.get('error')}) at TCP z {1000*pz[2]:.1f} mm "
                    f"(target {1000*xyz[2]:.1f}; part bottom {1000*(pz[2]-tcp_to_bottom_m):.1f}, base top "
                    f"{1000*(bp[2]+top):.1f})")
                _move(above, sq); return None, dict(out, error=f'{axis}: {lab} failed')
        r = _move(tcp_at(s_end), sq, max_force=MAX_FORCE_N, min_duration=(CLEAR_M + OVERTRAVEL_M) / SPEED_M_S)
        pc, _ = _tcp()
        s_contact = Rb.inv().apply(pc - st)                          # part shift from its seat at contact (base frame)
        chained_face = part_face + s_contact[k]
        err = float(face - chained_face)                             # real face - chained face along the axis
        out[axis] = {'contact': bool(r.get('contact')), 'err_mm': round(1000 * err, 2)}
        say(f"  [face-touch] {obj}: {'+y' if axis == 'y' else '+x'} face on the base's "
            f"{'-y' if axis == 'y' else '-x'} face {'(contact)' if r.get('contact') else '(NO contact)'}: "
            f"part {1000 * err:+.1f} mm from its chained pose along base {axis}")
        _move(tcp_at(s_start), sq)                                   # back off, then up
        _move(above, sq)
        if not r.get('contact') or abs(err) > MAX_ERR_M:
            return None, dict(out, error=f'{axis}: no clean contact')
    d_base = np.array([out['x']['err_mm'], out['y']['err_mm'], 0.0]) / 1000.0
    d_world = Rb.apply(d_base)
    return d_world[:2], dict(out, d_world_mm=[round(1000 * v, 2) for v in d_world[:2]])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--object-name', required=True); ap.add_argument('--base-name', required=True)
    ap.add_argument('--base-pose', type=float, nargs=7, required=True, help='x y z qx qy qz qw')
    ap.add_argument('--held-quat', type=float, nargs=4, required=True); ap.add_argument('--grasp-id', type=int, default=1)
    a = ap.parse_args()
    d, info = measure(a.object_name, a.base_name, (a.base_pose[:3], a.base_pose[3:]), a.held_quat, a.grasp_id)
    print(json.dumps(info))
