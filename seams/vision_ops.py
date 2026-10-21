#!/usr/bin/env python3
"""The vision layer's operations, one JSON answer per call (2026-10-20, M3a). No motion: every read is taken from
wherever the camera is; choosing and flying to a view is the planner's job (ARCHITECTURE_SEAMS: vision owns poses,
the held read and the certification measurement, never planning decisions or motion).

    python3 -m seams.vision_ops pose --name base1 --kind base [--timeout 15]
    python3 -m seams.vision_ops pose --name u_brown --kind part
    python3 -m seams.vision_ops certify --mate fmb1/mate:00

The last stdout line is `__VISION__ {json}`. seams.vision_server runs these per request (each read starts its own
rclpy context, so they cannot share the server's process).

What it wraps (the code the replay has used on every run, unchanged): the still-TCP pose observation
(observe_object_orientation: a pose counts only if the TCP was still 0.8 s before it was published), the base read
per BASE_FROM (markers -> depth outline -> 6-dof CAD fit, choose_base_read), the depth seat check
(seat_check.capture_world + measure: arrival-time stillness, heights against the base-top plane) and the marker
assembly check (queries.verify_assembly --xy-only, symmetry-aware yaw).
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]
os.environ.setdefault('BASE_FROM', 'icp')

STILL_WINDOW_S = 0.8            # the observe helper's acceptance rule (replay _OBSERVE_ORIENTATION_HELPER)
B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def resolve(mate):
    """'fmb1/mate:00' (or a bare 'mate:00' when one assembly is bound) -> (assembly, address, binding)."""
    asm, _, addr = mate.rpartition('/')
    bound = [k for k in B if k != 'sigma']
    if not asm:
        if len(bound) != 1:
            raise LookupError('assembly_ambiguous')
        asm = bound[0]
    if asm not in bound:
        raise LookupError('assembly_unbound')
    m = B[asm]['mates'].get(addr)
    if m is None:
        raise LookupError('mate_unknown')
    return asm, addr, dict(m, reference=B[asm]['reference'], acceptance=B[asm]['acceptance'],
                           order_position=B[asm]['order'].index(addr))


def _sym(name):
    try:
        from primitives.shared.config import get_symmetry_dir
        p = Path(get_symmetry_dir()) / f'{name}_symmetry.json'
        return p.stem if p.exists() else ''
    except Exception:
        return ''


def pose(name, kind, timeout=15.0):
    import replay_real_assembly as rr
    t_req = time.time()
    if kind == 'held':
        # the held read needs its own view motion (held_yaw_calibration) and is log-only today (HELD_YAW_APPLY False)
        return {'ok': False, 'error': 'held: not served yet - the planner takes the held read (seam open)'}
    if kind == 'base':
        ob = rr.observe_base_pose(name, timeout_s=timeout)
        if ob is None:
            return {'ok': False, 'error': f'{name}: not seen from this view'}
        nd = None
        try:
            from base_depth import measure as base_depth
            nd, info = base_depth(ob, name); log('[vision] depth outline', json.dumps(info))
        except Exception as e:
            log('[vision] no depth outline:', e)
        ob = rr.choose_base_read(name, ob, nd, say=log, tag='vision')
        src = rr.choose_base_read.source
        if src != 'icp':
            # the marker / outline base reads come back as one median with no spread kept: no sigma -> refused
            return {'ok': False, 'error': f'{name}: only a {src} read from this view (no stated sigma) - '
                                          f'the depth CAD fit needs the whole base in view'}
        s_xy = B['sigma']['base_icp']['xy']
        sigma = [s_xy, s_xy, math.nan, math.nan]       # z and yaw repeatability of the fit not measured yet
        source = 'depth_cad'
        (x, y, z), q = ob; n = 0
    elif kind == 'part':
        obs = rr.observe_object_orientation(name, timeout_s=timeout, n_samples=5)
        S = obs.get('samples') or []
        if obs.get('error') or not S:
            return {'ok': False, 'error': f"{name}: {obs.get('error') or 'not seen (still camera) from this view'}"}
        if len(S) < 3:
            return {'ok': False, 'error': f'{name}: {len(S)} still samples - too few to state a sigma'}
        P = np.array([s['xyz'] for s in S])
        x, y, z = np.median(P, axis=0)
        Q = np.array([s['quat_xyzw'] for s in S]); Q *= np.sign(Q @ Q[0])[:, None]
        q = Q.mean(0); q /= np.linalg.norm(q)
        from scipy.spatial.transform import Rotation as R
        yaws = np.unwrap([R.from_quat(qq).as_euler('xyz')[2] for qq in Q])
        # the spread of the still samples: repeatability of this read, not its bias (bias is the M2 tests' job)
        sigma = [float(v) for v in P.std(0, ddof=1)] + [float(np.std(yaws, ddof=1))]
        source = 'markers'; n = len(S)
    else:
        return {'ok': False, 'error': f'unknown kind {kind!r} (base | part | held)'}
    snap = rr.snapshot_robot_state(); tcp = (snap.get('tcp') or {}).get('xyz') or [math.nan] * 3
    return {'ok': True, 'name': name, 'frame_id': 'base', 'stamp': time.time(),
            # every accepted sample had the TCP still for STILL_WINDOW_S before it: a lower bound on stillness
            'stationary_since': t_req - STILL_WINDOW_S,
            'xyz': [float(x), float(y), float(z)], 'quat_xyzw': [float(v) for v in q],
            'sigma': sigma, 'source': source,
            'view_z': float(tcp[2]), 'detection_set': '', 'n': int(n), 'symmetry': _sym(name)}


def certify(mate):
    return certify_many([mate])[0]


def certify_many(mates):
    """Each mover's pose relative to its declared seat, from where the camera is: one base read and one still depth
    capture shared by all (execute_mate re-reads every predecessor NOW). z and the long-axis tilt from the depth seat
    check; xy from the marker check, reported only (see the judgement below). z not measured -> not certified
    (engagement_invalid upstream)."""
    import replay_real_assembly as rr
    import seat_check as sc
    R_ = [resolve(m_) for m_ in mates]
    bases = {m['reference'] for _, _, m in R_}
    obs, bev = {}, {}
    for base in bases:
        b = pose(base, 'base', timeout=8.0)
        if b['ok']:
            obs[base] = (tuple(b['xyz']), tuple(b['quat_xyzw'])); bev[base] = {'source': b['source'], 'xyz': b['xyz']}
        else:
            obs[base] = rr._last_base_pose(base); bev[base] = {'source': 'last saved read', 'why': b['error']}
    cap = sc.capture_world() if any(v is not None for v in obs.values()) else None
    out = []
    for asm, addr, m in R_:
        obj, base = m['mover'], m['reference']
        acc = m['acceptance']; sz = B['sigma']['seat_depth']; ob = obs[base]
        ev = {'mate': f'{asm}/{addr}', 'mover': obj, 'reference': base, 'order_position': m['order_position'],
              'declared': acc, 'base': bev[base]}
        z_err = tilt = math.nan
        if ob is None:
            ev['depth'] = {'error': 'no base pose'}
        elif cap is None:
            ev['depth'] = {'error': 'no still depth frames'}
        else:
            r = sc.measure(obj, ob, cap[0], base)
            ev['depth'] = {k: r.get(k) for k in ('ok', 'error', 'height_err_mm', 'pitch_deg', 'roll_deg', 'end_heights_mm',
                                                 'n_points', 'mode')}
            if r.get('height_err_mm') is not None:
                z_err = r['height_err_mm'] / 1000.0
                tilt = abs(r['pitch_deg']) if r.get('pitch_deg') is not None else math.nan
        rr.RUN_BASE['name'] = base
        rr.VERIFY_MAX_READS = 1          # xy is reported, not judged: one marker read (Speed T1; up to 8 cost ~10 s)
        v = rr._read_part(obj)
        ex = ey = math.nan
        if not rr._not_detected(v) and isinstance(v.get('position_error_m'), dict):
            ex, ey = float(v['position_error_m']['x']), float(v['position_error_m']['y'])
            ev['markers'] = {'position_error_m': v['position_error_m'], 'orientation_error_deg': v.get('orientation_error_deg')}
        else:
            ev['markers'] = {'error': v.get('error') or 'not detected'}
        # judged on what carries a stated sigma: the seat height (depth, test-measured repeatability). Marker xy has no
        # stated sigma: reported, not judged - for an insertion mate a mover at seat depth is inside its pocket.
        # Tilt is judged in the unit that has a stated sigma: the heights at both ends of the top face against the same
        # z band; the angle is reported only (2026-10-21: one seated u_brown read -2.9 / -1.9 / -1.6 deg from three
        # views - the angle's repeatability is not measured, so it cannot decide).
        lim_z = acc['halfwidth_m'][2] + 2 * sz['z']
        ev['limits'] = {'z_m': lim_z, 'rule': 'centre and both end heights: declared half-width + 2 sigma (seat_depth); '
                                              'tilt angle and marker xy: reported only (no stated sigma)'}
        xy_seen = math.isfinite(ex)
        ends = (ev.get('depth') or {}).get('end_heights_mm') or []
        if math.isfinite(z_err) and ends and min(ends) / 1000.0 < -lim_z:
            # the pocket floor stops the part: an end read below the seat by more than the band is the view, not
            # the part (2026-10-21: u_brown seen across the base read -2.29 mm at one end) -> not measured
            ev['invalid'] = f'an end {min(ends):.2f} mm below the seat - physically impossible, reading rejected'
            z_err = tilt = math.nan
        if not math.isfinite(z_err):
            inside, method = False, 'markers' if xy_seen else 'none'
            ev['verdict'] = 'not certified: ' + (ev.get('invalid') or 'seat height not measured from this view')
        else:
            inside = bool(abs(z_err) <= lim_z and all(abs(e) / 1000.0 <= lim_z for e in ends))
            method = 'both' if xy_seen else 'depth_seat_check'
            ev['verdict'] = ('inside' if inside else 'outside') + ' (xy implied by seat depth)'
        out.append({'mate_id': f'{asm}/{addr}', 'inside': inside, 'error_mate_frame_m': [ex, ey, z_err],
                    'tilt_deg': tilt, 'method': method, 'evidence': ev})
    return out


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='op', required=True)
    p = sp.add_parser('pose'); p.add_argument('--name', required=True); p.add_argument('--kind', required=True)
    p.add_argument('--timeout', type=float, default=15.0)
    c = sp.add_parser('certify'); c.add_argument('--mate', required=True, nargs='+')
    a = ap.parse_args()
    try:
        if a.op == 'pose':
            out = pose(a.name, a.kind, a.timeout)
        else:
            out = certify(a.mate[0]) if len(a.mate) == 1 else {'ok': True, 'results': certify_many(a.mate)}
    except LookupError as e:
        out = {'ok': False, 'error': str(e), 'refusal': str(e)}
    except Exception as e:
        out = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    print('__VISION__ ' + json.dumps(out, default=float), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
