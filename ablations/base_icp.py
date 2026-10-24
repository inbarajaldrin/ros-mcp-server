#!/usr/bin/env python3
"""Base pose from depth by fitting the CAD model to the point cloud (2026-10-20, operator: "if the depth is
used to localize an object with cad and pointcloud you will get much better pose").

The model is the base mesh plus every part seated so far at its CAD seat pose (more surfaces than the top
square the outline fit uses: slot walls, pockets, the seated parts). Depth points near the base (above the
table) are matched to their nearest model point (data -> model, so a partial view still fits) and the base
pose is solved in x, y, yaw (+ a z offset that absorbs depth bias; roll/pitch stay flat on the table) by
trimmed iterative closest point, coarse (15 mm gate) then fine (4 mm gate).

    python3 -u ablations/base_icp.py --base base3 --seated line_brown line_red fork_orange fork_yellow \
        --views top tilt+x tilt-x tilt+y        # moves the arm (camera views of the base), prints the spread
    python3 -u ablations/base_icp.py --base base3 --no-move                    # fit from where the arm is
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))


import functools


def model_points(base, seated, step=0.0015):
    return _model_points(base, tuple(seated), step)


@functools.lru_cache(maxsize=16)
def _model_points(base, seated, step=0.0015):
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    V, F = load_mesh(base); P = [surface_points(V, F, step)]
    for p in seated:
        V, F = load_mesh(p); sp, sq = load_seat_pose_in_base(base, p)
        P.append(surface_points(V, F, step) @ R.from_quat(sq).as_matrix().T + sp)
    M = np.vstack(P)
    return M[M[:, 2] > M[:, 2].min() + 0.003]          # drop the base's bottom face (never seen)


def fit(base, seated, Pw, init_ob, table_z=None):
    """Returns (((x, y, z), quat), info)."""
    M = model_points(base, seated); tree = cKDTree(M)
    (bx, by, bz), bq = init_ob
    yaw = float(R.from_quat(bq).as_euler('xyz')[2]); t = np.array([bx, by]); dz = 0.0
    if table_z is None:
        table_z = float(np.percentile(Pw[:, 2], 30))
    near = (np.hypot(Pw[:, 0] - bx, Pw[:, 1] - by) < 0.13) & (Pw[:, 2] > table_z + 0.004) & (Pw[:, 2] < table_z + 0.12)
    D = Pw[near]
    if len(D) > 60000:
        D = D[np.random.default_rng(0).choice(len(D), 60000, replace=False)]
    info = {'n_data': int(len(D))}
    for gate, iters in ((0.015, 15), (0.004, 25)):
        for _ in range(iters):
            c, s = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s], [s, c]])
            Db = np.c_[(D[:, :2] - t) @ Rz, D[:, 2] - bz - dz]            # data in the base frame
            d, idx = tree.query(Db, distance_upper_bound=gate)
            ok = np.isfinite(d)
            if ok.sum() < 500:
                return None, dict(info, error=f'only {int(ok.sum())} matches at {1000*gate:.0f} mm')
            A = D[ok, :2]; B = M[idx[ok], :2]                              # world xy  <->  model xy (base frame)
            # solve world = R(yaw) * model + t  (2D Kabsch)
            ca, cb = A.mean(0), B.mean(0)
            H = (B - cb).T @ (A - ca)
            U, S_, Vt = np.linalg.svd(H)
            Rn = Vt.T @ U.T
            if np.linalg.det(Rn) < 0:
                Vt[1] *= -1; Rn = Vt.T @ U.T
            yaw = float(np.arctan2(Rn[1, 0], Rn[0, 0])); t = ca - Rn @ cb
            dz = float(np.median(D[ok, 2] - bz - M[idx[ok], 2]))
    c, s = np.cos(yaw), np.sin(yaw)
    Db = np.c_[(D[:, :2] - t) @ np.array([[c, -s], [s, c]]), D[:, 2] - bz - dz]
    d, _ = tree.query(Db, distance_upper_bound=0.004)
    ok = np.isfinite(d)
    info.update(inliers=int(ok.sum()), inlier_frac=round(float(ok.mean()), 3), rms_mm=round(1000 * float(np.sqrt(np.mean(d[ok] ** 2))), 2),
                dz_mm=round(1000 * dz, 2), xy_mm=[round(1000 * t[0], 1), round(1000 * t[1], 1)], yaw_deg=round(np.degrees(yaw), 2))
    return ((float(t[0]), float(t[1]), float(bz)), R.from_euler('z', yaw).as_quat()), info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', required=True)
    ap.add_argument('--seated', nargs='*', default=[])
    ap.add_argument('--views', nargs='*', default=['top'])
    ap.add_argument('--no-move', action='store_true')
    a = ap.parse_args()
    from seat_check import capture_world, X_CAM
    from replay_real_assembly import observe_base_pose
    from base_depth import measure as outline, VIEW_FLANGE_Z
    from tilted_seat_view import aimed_flange
    ob0 = observe_base_pose(a.base)
    if ob0 is None:
        sys.exit(f'{a.base} not observed')
    rows = []
    for v in (['here'] if a.no_move else a.views):
        if v != 'here':
            (bx, by, bz), bq = ob0
            if v == 'top':
                pf, qf = [bx, by, VIEW_FLANGE_Z], [0, 1, 0, 0]
            else:
                d = np.zeros(3); d['xy'.index(v[-1])] = 1.0 if v[-2] == '+' else -1.0
                pf, qf, _ = aimed_flange(np.array([bx, by, bz + 0.0175]), d, 25.0, 0.55, X_CAM, 0.0)
            r = subprocess.run(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                *[f"{x:.4f}" for x in pf], '--quat', *[f"{x:.6f}" for x in qf], '--min-z', '0.15'],
                               cwd=str(REPO), capture_output=True, text=True, timeout=90)
            if r.returncode != 0:
                print(f"{v}: move refused"); continue
            time.sleep(1.5)
        ob = observe_base_pose(a.base, timeout_s=6.0) or ob0
        cap = capture_world(n=12)
        if cap is None:
            print(f"{v}: no depth"); continue
        pose, info = fit(a.base, a.seated, cap[0], ob)
        od, oinfo = outline(ob, a.base)
        mk = (1000 * ob[0][0], 1000 * ob[0][1], float(R.from_quat(ob[1]).as_euler('xyz', degrees=True)[2]))
        ol = tuple(oinfo.get('centre_mm', [np.nan, np.nan])) + (oinfo.get('yaw_deg', np.nan),)
        rows.append(dict(view=v, icp=info, marker=[round(x, 2) for x in mk], outline=ol))
        print(json.dumps(rows[-1]), flush=True)
    if len(rows) >= 2:
        for k in ('icp', 'marker', 'outline'):
            P = np.array([(r['icp']['xy_mm'] + [r['icp']['yaw_deg']]) if k == 'icp' else list(r[k]) for r in rows
                          if (k != 'icp' or 'xy_mm' in r['icp'])], dtype=float)
            if len(P) >= 2:
                print(f"{k:8s} across views: xy spread max {np.max(np.hypot(*(P[:, :2] - np.median(P[:, :2], 0)).T)):.1f} mm, "
                      f"yaw range {np.ptp(P[:, 2]):.2f} deg  (median {np.round(np.median(P, 0), 1).tolist()})")
    return 0


if __name__ == '__main__':
    sys.exit(main())


# ---------------------------------------------------------------------------------------------------------
# 6-dof read in the camera frame (2026-10-20): the 4-dof fit above forces the base flat; base3 sits ~0.9 deg
# pitched in the robot frame (every 6-dof view agrees), and forcing it flat shifted xy by up to 11 mm in
# tilted views. Fitting all 6 dof in the camera frame and reporting x, y, yaw gave a 1.8 mm median spread
# over 11 views (top-down at two heights and three wrist yaws, tilted 20/30 deg) - the mount needs no refit.
# ---------------------------------------------------------------------------------------------------------
COVERAGE_MIN = 0.85     # matched points must span this fraction of the base in x and in y


TILT_MAX_DEG = 2.5            # a clean fit reads ~0.4 / 1.0 deg (the base is flat on the table)
Z_OFF_MAX_M = 0.003           # from the base's height on the table (DEFAULT_BASE_POSITION z)
N_MIN = 4000                  # clean fits match ~7800 of 8000 sampled points


def read_icp(base, ob, seated=(), n_points=8000, still_already=False):
    """((x, y, z_marker), quat_yaw), info from one still capture where the arm is; z is the marker read's
    (table-snapped, like every base read). None + info on failure."""
    from seat_check import X_CAM as X0
    from handeye_icp_survey import capture, fit6, T
    c = capture(still_already=still_already)
    if c is None:
        return None, {'error': 'no still depth capture'}
    d, K, tcp = c
    d = d.astype(np.float32)[::2, ::2] / 1000.0; K = K.copy(); K[:2] /= 2.0
    vv, uu = np.nonzero(d > 0.1); z = d[vv, uu]
    Pc = np.c_[(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
    Tw = T(tcp[:3], tcp[3:]) @ X0
    Pw = Pc @ Tw[:3, :3].T + Tw[:3, 3]
    (bx, by, bz), bq = ob
    table = np.percentile(Pw[:, 2], 30)
    m = (np.hypot(Pw[:, 0] - bx, Pw[:, 1] - by) < 0.13) & (Pw[:, 2] > table + 0.004) & (Pw[:, 2] < table + 0.12)
    idx = np.nonzero(m)[0]
    if len(idx) < 2000:
        return None, {'error': f'only {len(idx)} points near the base'}
    Pc = Pc[np.random.default_rng(0).choice(idx, min(len(idx), n_points), replace=False)]
    M = model_points(base, list(seated)); tree = cKDTree(M)
    Tcb, rms, n = fit6(Pc, M, tree, np.linalg.inv(Tw) @ T([bx, by, bz], bq))
    if Tcb is None:
        return None, {'error': f'fit failed ({n} matches)'}
    Bw = Tw @ Tcb
    # coverage: the matched points must span the base's footprint in BOTH axes of its own frame; a partial view
    # leaves the fit free to slide along base3's channels (2026-10-20: two chained fits walked 9 + 12 mm in y with the
    # base half out of the depth image)
    Tbc = np.linalg.inv(Tcb)
    Pb = Pc @ Tbc[:3, :3].T + Tbc[:3, 3]
    d, _ = tree.query(Pb, distance_upper_bound=0.003); okm = np.isfinite(d)
    ext = M[:, :2].max(0) - M[:, :2].min(0)
    span = (np.percentile(Pb[okm, :2], 98, axis=0) - np.percentile(Pb[okm, :2], 2, axis=0)) if okm.sum() > 50 else np.zeros(2)
    cov = span / ext
    e = R.from_matrix(Bw[:3, :3]).as_euler('xyz', degrees=True)
    yaw = float(np.degrees(np.arctan2(Bw[1, 0], Bw[0, 0])))
    info = {'rms_mm': round(rms, 2), 'n': n, 'xy_mm': [round(1000 * Bw[0, 3], 1), round(1000 * Bw[1, 3], 1)],
            'yaw_deg': round(yaw, 2), 'roll_pitch_deg': [round(e[0], 2), round(e[1], 2)],
            'z_mm': round(1000 * Bw[2, 3], 1), 'marker_mm': [round(1000 * bx, 1), round(1000 * by, 1)]}
    info['coverage'] = [round(float(c), 2) for c in cov]
    if rms > 2.5 or n < 1500:
        return None, dict(info, error='fit too loose')
    if (cov < COVERAGE_MIN).any():
        return None, dict(info, error=f'partial view: matched points span {np.round(cov, 2).tolist()} of the base')
    # the base lies flat on the table: a fit tilted or lifted off it locked onto something else (10-22 base-read audit:
    # in-hand fits with line_green held - roll/pitch ~4/11 deg, z -50..-54 mm vs -62.5, n 816-2300 - landed 7.6-13.7 mm
    # off the markers; two passed rms + coverage and seeded the next reads)
    if max(abs(e[0]), abs(e[1])) > TILT_MAX_DEG:
        return None, dict(info, error=f'tilted fit (roll/pitch {e[0]:+.1f}/{e[1]:+.1f} deg) - the base lies flat')
    # against the base's height on the table, not the seed's: an in-hand seed is a marker read 4-5 mm high, and 21 good
    # in-hand fits (rms ~1.1 mm, n ~7800, flat, z -62.6..-62.9) were rejected against it (10-23, run 31: ~25 s each)
    from primitives.shared.config import DEFAULT_BASE_POSITION
    z_tab = float(DEFAULT_BASE_POSITION[2])
    if abs(Bw[2, 3] - z_tab) > Z_OFF_MAX_M:
        return None, dict(info, error=f'fit {1000 * (Bw[2, 3] - z_tab):+.1f} mm off the table height')
    if n < N_MIN:
        return None, dict(info, error=f'only {n} matched points')
    return ((float(Bw[0, 3]), float(Bw[1, 3]), float(bz)), R.from_euler('z', yaw, degrees=True).as_quat()), info
