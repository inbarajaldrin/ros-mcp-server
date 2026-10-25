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


FACE_DOWN = [R.identity(), R.from_euler('x', 90, degrees=True), R.from_euler('x', 180, degrees=True),
             R.from_euler('x', -90, degrees=True), R.from_euler('y', 90, degrees=True), R.from_euler('y', -90, degrees=True)]


def fit_any_stance(part, Pw, seed, table_z):
    """The fit without the marker's tilt: every resting face of the CAD (6 face-down rotations) x 4 starting yaws about
    the marker's, solved by fit(); the lowest-rms one wins (operator 10-24: depth + CAD tell you exactly what you are
    looking at - a marker read from another face gave line_green a tilt 8.9 deg off its depth fit). Returns (pose, info)
    with 'stance': the winner's face index and how far the marker's orientation is from it (deg)."""
    (sx, sy, sz), sq = seed
    yaw_m = float(R.from_quat(sq).as_euler('xyz')[2])
    # choose on a quarter of the points near the seed (24 full fits took 35 s), then refit the winner on all of them
    near = (np.hypot(Pw[:, 0] - sx, Pw[:, 1] - sy) < CROP_R_M) & (Pw[:, 2] > table_z + ABOVE_TABLE_M)
    Pn = Pw[near]
    Ps = Pn[np.random.default_rng(0).choice(len(Pn), min(len(Pn), max(N_MIN * 4, len(Pn) // 4)), replace=False)] \
        if len(Pn) > N_MIN * 4 else Pn
    best = (None, {'error': 'no stance fitted'})
    for fi, F in enumerate(FACE_DOWN):
        for k in range(4):
            q0 = (R.from_euler('z', yaw_m + k * np.pi / 2) * F).as_quat()
            pose, info = fit(part, Ps, ((sx, sy, sz), tuple(q0)), table_z)
            if pose is not None and (best[0] is None or info['rms_mm'] < best[1]['rms_mm']):
                best = (pose, dict(info, stance={'face': fi, 'yaw_start_deg': round(np.degrees(yaw_m) + 90 * k, 1)}))
    if best[0] is not None:
        st_ = best[1]['stance']
        pose, info = fit(part, Pw, best[0], table_z)
        best = (pose, dict(info, stance=st_)) if pose is not None else best
    if best[0] is not None:
        d = (R.from_quat(best[0][1]) * R.from_quat(sq).inv()).magnitude()
        best[1]['stance']['marker_off_deg'] = round(float(np.degrees(d)), 2)
    return best


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


CAL_LOG = REPO / 'ablations' / 'logs' / 'marker_vs_depth.jsonl'   # marker pose next to the depth+CAD pose (per-face calibration)


def read_any(part, seed=None, timeout=8.0):
    """As read(), with the stance chosen by depth + CAD (fit_any_stance), not taken from the marker; logs the marker
    pose next to the depth one to CAL_LOG for a per-face marker calibration."""
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
    pose, info = fit_any_stance(part, Pw, seed, table_z)
    info['table_z_mm'] = round(1000 * table_z, 1)
    try:
        with open(CAL_LOG, 'a') as fh:
            fh.write(json.dumps({'t': time.time(), 'part': part, 'marker': [list(seed[0]), list(seed[1])],
                                 'depth': [list(pose[0]), list(pose[1])] if pose else None,
                                 'rms_mm': info.get('rms_mm'), 'stance': info.get('stance')}) + '\n')
    except Exception:
        pass
    return pose, info


SIX_AGREE_MM, SIX_AGREE_DEG, SIX_MIN_AGREE = 2.0, 2.0, 3   # 6-dof starts that must converge on one pose


def read_6dof(part, timeout=8.0, others=('u_brown', 'u_orange', 'line_green', 'inverted_u_yellow')):
    """A part lying ON or AGAINST another one (tilted, not on the table): each neighbour the markers see is fitted from
    its marker + CAD and its points removed, then `part` is fitted in 6 dof from its own marker and the marker's IPPE
    flips. Trusted when >= SIX_MIN_AGREE starts converge (operator 10-25: "is markers cad and depth not enough" -
    u_brown on the tipped yellow: flat fits flipped 15 deg, unseeded 6-dof starts spread 32 mm; with yellow's points
    removed and the marker seed, 7 starts agreed to 0.4 mm / 0.8 deg at rms 1.08 mm). (pose | None, info)."""
    from handeye_icp_survey import capture, T, fit6
    from seat_check import X_CAM
    from replay_real_assembly import observe_many
    names = [part] + [o for o in others if o != part]
    S = observe_many(names, timeout, n_samples=8)
    if not S.get(part):
        return None, {'error': f'{part}: no marker read here'}
    seeds = {n: (tuple(np.median(np.array([s['xyz'] for s in S[n]]), axis=0)), tuple(S[n][-1]['quat_xyzw']))
             for n in names if S.get(n)}
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
    (sx, sy, sz), sq = seeds[part]
    D = Pw[(np.hypot(Pw[:, 0] - sx, Pw[:, 1] - sy) < CROP_R_M) & (Pw[:, 2] > table_z + ABOVE_TABLE_M)
           & (Pw[:, 2] < table_z + 0.10)]
    info = {'n_data': int(len(D)), 'table_z_mm': round(1000 * table_z, 1), 'removed': {}}
    for n, sd in seeds.items():
        if n == part or np.hypot(sd[0][0] - sx, sd[0][1] - sy) > CROP_R_M + 0.06:
            continue
        pn, inf_n = fit(n, Pw, sd, table_z)
        if pn is None:
            continue
        Mn = R.from_quat(pn[1]).apply(model_points(n)) + np.array(pn[0])
        dn, _ = cKDTree(Mn).query(D, distance_upper_bound=0.004)
        info['removed'][n] = int(np.isfinite(dn).sum()); D = D[~np.isfinite(dn)]
    if len(D) < N_MIN:
        return None, dict(info, error='too few depth points left for the part')
    if len(D) > 40000:
        D = D[np.random.default_rng(0).choice(len(D), 40000, replace=False)]
    M = model_points(part); tree = cKDTree(M)
    q = R.from_quat(sq); e = q.as_euler('xyz', degrees=True)
    starts = [q, R.from_euler('xyz', [e[0], -e[1], e[2]], degrees=True),
              R.from_euler('xyz', [-e[0], e[1], e[2]], degrees=True), R.from_euler('z', 180, degrees=True) * q]
    res = []
    for qs in starts:
        for dp in ((0, 0, 0), (0.003, 0, 0), (0, 0.003, 0), (0, 0, 0.003)):
            T0 = np.eye(4); T0[:3, :3] = qs.as_matrix(); T0[:3, 3] = np.array([sx, sy, sz]) + dp
            Tf, rms, n = fit6(D, M, tree, T0)
            if Tf is not None:
                res.append((rms, n, Tf))
    if not res:
        return None, dict(info, error='6-dof fit lost the part from every start')
    best = min(res, key=lambda r: r[0])
    agree = [r for r in res if 1000 * np.linalg.norm(r[2][:3, 3] - best[2][:3, 3]) <= SIX_AGREE_MM
             and np.degrees((R.from_matrix(r[2][:3, :3]) * R.from_matrix(best[2][:3, :3]).inv()).magnitude()) <= SIX_AGREE_DEG]
    Tb = best[2]; qb = R.from_matrix(Tb[:3, :3])
    tilt = float(np.degrees(np.arccos(np.clip(abs((qb * R.from_quat(sq).inv()).apply([0, 0, 1])[2]), -1, 1))))
    info.update(rms_mm=round(best[0], 2), n=int(best[1]), starts=len(res), agree=len(agree),
                xy_mm=[round(1000 * Tb[0, 3], 1), round(1000 * Tb[1, 3], 1)], z_mm=round(1000 * Tb[2, 3], 1),
                moved_mm=[round(1000 * (Tb[0, 3] - sx), 1), round(1000 * (Tb[1, 3] - sy), 1)],
                rpy_deg=[round(float(v), 1) for v in qb.as_euler('xyz', degrees=True)], to_marker_tilt_deg=round(tilt, 1))
    pose = ((float(Tb[0, 3]), float(Tb[1, 3]), float(Tb[2, 3])), tuple(float(v) for v in qb.as_quat()))
    if best[0] > RMS_MAX_MM or len(agree) < SIX_MIN_AGREE:
        return None, dict(info, error=f'6-dof starts did not converge ({len(agree)} agree) or loose fit')
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
