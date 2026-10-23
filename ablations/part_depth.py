#!/usr/bin/env python3
"""A part lying on the table, located by fitting its CAD mesh to the depth cloud (2026-10-21).

    python3 -u ablations/part_depth.py --part line_green            # from where the arm is (no motion)

Operator 2026-10-21, after a finger landed on line_green (marker read 10 mm off where the disassembly had set it
down): "rely on depth to find the issue and fix perception". Same scheme as the base read (base_icp): the marker
pose seeds the fit; depth points around it and above the table are matched to their nearest model point (data ->
model, so the hidden faces do not matter) and x, y, yaw (+ a z offset that absorbs depth bias) are solved by trimmed
ICP, coarse then fine; roll/pitch stay at the marker stance (a part rests flat on one face). Prints the fit and the
marker read side by side.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]

CROP_R_M = 0.11           # depth points this close to the seed (xy) ...
ABOVE_TABLE_M = 0.004     # ... and this far above the table plane belong to the part (or a neighbour)
RMS_MAX_MM, N_MIN = 2.0, 400


def model_points(part, step=0.0015):
    from grasp_depth import load_mesh, surface_points
    V, F = load_mesh(part)
    return surface_points(V, F, step)


def fit(part, Pw, seed, table_z):
    """Fit `part` (seeded at ((x, y, z), quat)) to world points Pw. Returns (pose, info) or (None, info)."""
    (sx, sy, sz), sq = seed
    near = (np.hypot(Pw[:, 0] - sx, Pw[:, 1] - sy) < CROP_R_M) & (Pw[:, 2] > table_z + ABOVE_TABLE_M) \
        & (Pw[:, 2] < table_z + 0.10)
    D = Pw[near]
    info = {'n_data': int(len(D))}
    if len(D) < N_MIN:
        return None, dict(info, error='too few depth points around the seed')
    if len(D) > 40000:
        D = D[np.random.default_rng(0).choice(len(D), 40000, replace=False)]
    M0 = model_points(part)
    Rs = R.from_quat(sq)                                  # stance (roll/pitch kept), yaw solved below
    yaw0 = float(Rs.as_euler('xyz')[2]); Rtilt = R.from_euler('z', -yaw0) * Rs
    t = np.array([sx, sy]); yaw = yaw0; dz = float(sz)
    for gate, iters in ((0.015, 20), (0.006, 20), (0.003, 25)):
        for _ in range(iters):
            Rw = R.from_euler('z', yaw) * Rtilt
            Mw = M0 @ Rw.as_matrix().T + np.array([t[0], t[1], dz])
            tree = cKDTree(Mw)
            d, idx = tree.query(D, distance_upper_bound=gate)
            ok = np.isfinite(d)
            if ok.sum() < N_MIN // 2:
                return None, dict(info, error=f'fit lost the part ({int(ok.sum())} matches)')
            A, B = Mw[idx[ok]], D[ok]                       # model -> data, in 2-D + z offset
            ca, cb = A[:, :2].mean(0), B[:, :2].mean(0)
            H = (A[:, :2] - ca).T @ (B[:, :2] - cb)
            dth = float(np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1]))
            c, s = np.cos(dth), np.sin(dth)
            Rz = np.array([[c, -s], [s, c]])
            t = Rz @ (t - ca) + cb                           # rotate the model about its matched centroid
            yaw += dth; dz += float(np.median(B[:, 2] - A[:, 2]))
            if abs(dth) < 1e-5:
                break
    rms = float(np.sqrt(np.mean(d[ok] ** 2)))
    pose = ((float(t[0]), float(t[1]), float(dz)), tuple(float(v) for v in (R.from_euler('z', yaw) * Rtilt).as_quat()))
    info.update(rms_mm=round(1000 * rms, 2), n=int(ok.sum()), xy_mm=[round(1000 * t[0], 1), round(1000 * t[1], 1)],
                yaw_deg=round(float(np.degrees(yaw)), 2), seed_xy_mm=[round(1000 * sx, 1), round(1000 * sy, 1)],
                seed_yaw_deg=round(float(np.degrees(yaw0)), 2),
                moved_mm=[round(1000 * (t[0] - sx), 1), round(1000 * (t[1] - sy), 1)],
                turned_deg=round(float(np.degrees(yaw - yaw0)), 2))
    if rms * 1000 > RMS_MAX_MM or ok.sum() < N_MIN:
        return None, dict(info, error='fit too loose')
    return pose, info


def read(part, seed=None, timeout=8.0):
    """From where the arm is: still capture, marker seed (unless given), fit. (pose | None, info)."""
    from handeye_icp_survey import capture, T
    from seat_check import X_CAM
    if seed is None:
        from replay_real_assembly import observe_many
        S = observe_many([part], timeout, n_samples=8).get(part) or []
        if not S:
            return None, {'error': f'{part}: no marker read here'}
        P = np.median(np.array([s['xyz'] for s in S]), axis=0)
        seed = (tuple(P), tuple(S[-1]['quat_xyzw']))
    c = capture()
    if c is None:
        return None, {'error': 'no still depth capture'}
    d, K, tcp = c
    d = d.astype(np.float32) / 1000.0
    vv, uu = np.nonzero(d > 0.1); z = d[vv, uu]
    Pc = np.c_[(uu - K[0, 2]) / K[0, 0] * z, (vv - K[1, 2]) / K[1, 1] * z, z]
    Tw = T(tcp[:3], tcp[3:]) @ X_CAM
    Pw = Pc @ Tw[:3, :3].T + Tw[:3, 3]
    table_z = float(np.percentile(Pw[:, 2], 30))
    pose, info = fit(part, Pw, seed, table_z)
    info['table_z_mm'] = round(1000 * table_z, 1)
    return pose, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', required=True)
    a = ap.parse_args()
    pose, info = read(a.part)
    print(json.dumps(dict(info, ok=pose is not None)), flush=True)
    return 0 if pose is not None else 1


if __name__ == '__main__':
    sys.exit(main())
