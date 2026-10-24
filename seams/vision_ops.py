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


SEED_MAX_AGE_S = 900.0          # a saved base read this recent seeds the certify's CAD fit


def pose(name, kind, timeout=15.0):
    import replay_real_assembly as rr
    t_req = time.time()
    if kind == 'held':
        # the held read needs its own view motion (held_yaw_calibration) and is log-only today (HELD_YAW_APPLY False)
        return {'ok': False, 'error': 'held: not served yet - the planner takes the held read (seam open)'}
    if kind == 'base':
        # only the depth CAD fit carries a stated sigma, so it is the read: seeded by the last saved base read when it is
        # recent (the base moves mm between inserts; the fit's own checks reject a bad convergence), by the markers
        # otherwise. A certify spent 15.2 s here (10-23): markers 5.5 + a depth outline only a refused fallback used 3.3
        # + the fit 4.3
        from base_icp import read_icp
        oi, info, seed_src = None, {}, None
        try:
            p_, q_, t_ = json.load(open(rr.LAST_BASE_POSE))[name]
            if time.time() - float(t_) < SEED_MAX_AGE_S:
                oi, info = read_icp(name, (tuple(p_), tuple(q_))); seed_src = 'last read'
                if oi is None:
                    log('[vision] CAD fit from the last read failed:', json.dumps(info, default=str))
            else:
                log(f'[vision] saved base read {time.time() - float(t_):.0f} s old - markers seed the fit')
        except Exception as e:
            log(f'[vision] no saved base read ({e})')
        if oi is None and seed_src is None:
            # no recent saved read: the markers seed it. A seeded fit that FAILED means this view does not hold the whole
            # base (a seat view): a marker seed fails the same way (+10 s), and the caller falls back to the saved read
            ob = rr.observe_base_pose(name, timeout_s=timeout)
            if ob is None:
                return {'ok': False, 'error': f'{name}: not seen from this view'}
            oi, info = read_icp(name, ob); seed_src = 'markers'
        log('[vision] CAD fit', seed_src, json.dumps(info, default=str))
        if oi is None:
            return {'ok': False, 'error': f"{name}: the depth CAD fit failed from this view ({info.get('error')}) - "
                                          f"it needs the whole base in view"}
        ob = oi
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


def _vol_centroid(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    v = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    return ((a + b + c) / 4.0 * v[:, None]).sum(0) / v.sum()


def graph_seat_dz(obj, base, graph_z_mm):
    """How far our assembly JSON's seat sits ABOVE the graph's declared seat (m), centroid to centroid. The
    certification reference top face is ours minus this."""
    if graph_z_mm is None:
        return 0.0
    from grasp_depth import load_mesh
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    from scipy.spatial.transform import Rotation as R
    Vb, Fb = load_mesh(base); cb = _vol_centroid(Vb, Fb)
    V, F = load_mesh(obj); sp, sq = load_seat_pose_in_base(base, obj)
    c = _vol_centroid(V @ R.from_quat(sq).as_matrix().T + sp, F)
    return float((c[2] - cb[2]) - graph_z_mm / 1000.0)


def certify(mate):
    return certify_many([mate])[0]


TAP_MAX_AGE_S = 1800.0


def tap_ends(obj, base):
    """The mover's end heights (mm vs the seated CAD top face, at the CAD ends) from its last tap map, or (None, why).
    Used only while nothing can have moved it since: no insert and no verdict on any part after the taps. Why: depth
    fits a plane to the visible top face and extrapolates it to the ends - on line_green (10-23, runs 31/32) it read
    +2.2 mm over contact at the centre and tilted -0.9 deg on a bar five taps put flat to 0.06 mm, so a seated bar's
    end read +2.47 mm against the +-2.0 band. Contact carries the smaller error; depth still judges the centre."""
    from datetime import datetime
    from seams.seated_parts import SeatedParts
    S = SeatedParts()
    e = S.entry(obj)
    t, m, al = e.get('tap_t'), e.get('tap_map_mm') or {}, e.get('tap_along_mm') or {}
    if t is None or not al:
        return None, 'no timed tap map'
    if time.time() - t > TAP_MAX_AGE_S:
        return None, f'tap map {time.time() - t:.0f} s old'
    later = [p for p, v in S.ledger().items() if p != obj and (v.get('t') or 0) > t]
    for p, v in S.seated_offsets().items():
        try:
            if datetime.strptime(v['t'], '%Y-%m-%d %H:%M:%S').timestamp() > t + 1.0:
                later.append(f'{p} inserted')
        except Exception:
            pass
    if later:
        return None, f'parts touched since the taps: {later}'
    pts = [(float(al[k]), float(m[k])) for k in ('mid', 'end-', 'end+')
           if isinstance(m.get(k), (int, float)) and k in al]
    if len(pts) < 3:
        return None, 'tap map lacks mid and both ends'
    import sys as _s
    _s.path.insert(0, str(REPO / 'ablations'))
    from seat_check import TOP_FACE
    hl = 1000 * float(TOP_FACE[(obj, base)]['half_long_m'])
    a = np.array(pts)
    c1, c0 = np.polyfit(a[:, 0], a[:, 1], 1)
    return [round(float(c0 - c1 * hl), 2), round(float(c0 + c1 * hl), 2)], f"taps {time.time() - t:.0f} s ago {m}"


INSERT_DEPTH_MAX_AGE_S = 900.0    # the insert's seat - prediction, recorded this recently, stands for a no-wedge mover


def _insert_depth(obj, lim_z):
    """z (mm) of the insert's recorded seat - prediction for `obj` when recent and inside +-lim_z, else None."""
    from datetime import datetime
    from seams.seated_parts import SeatedParts
    v = SeatedParts().seated_offsets().get(obj) or {}
    try:
        t = datetime.strptime(v['t'], '%Y-%m-%d %H:%M:%S').timestamp()
        z = float(v['seat_minus_pred_mm'][2])
    except Exception:
        return None
    return z if time.time() - t <= INSERT_DEPTH_MAX_AGE_S and abs(z) <= 1000 * lim_z else None


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
    T = {'t0': time.time()}                     # where a certify's ~20 s goes (stderr; TGOAL)

    def lap(k):
        T[k] = round(time.time() - T.pop('_last', T['t0']), 2); T['_last'] = time.time()
    for base in bases:
        b = pose(base, 'base', timeout=8.0)
        if b['ok']:
            obs[base] = (tuple(b['xyz']), tuple(b['quat_xyzw'])); bev[base] = {'source': b['source'], 'xyz': b['xyz']}
        else:
            obs[base] = rr._last_base_pose(base); bev[base] = {'source': 'last saved read', 'why': b['error']}
    lap('base_pose')
    cap = sc.capture_world() if any(v is not None for v in obs.values()) else None
    lap('capture')
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
            # against the graph's declared seat: our CAD reference is dz above it, so a part on the graph's seat reads
            # -dz against ours (yellow: -5.8 mm measured, dz +6.0 mm)
            dz = graph_seat_dz(obj, base, m.get('graph_seat_z_mm'))
            if r.get('height_err_mm') is not None:
                r['height_err_mm'] = round(r['height_err_mm'] + 1000 * dz, 2)
            if r.get('end_heights_mm'):
                r['end_heights_mm'] = [round(e + 1000 * dz, 2) for e in r['end_heights_mm']]
            ev['graph_seat_dz_mm'] = round(1000 * dz, 2)
            ev['depth'] = {k: r.get(k) for k in ('ok', 'error', 'height_err_mm', 'pitch_deg', 'roll_deg', 'end_heights_mm',
                                                 'n_points', 'mode')}
            if r.get('height_err_mm') is not None:
                z_err = r['height_err_mm'] / 1000.0
                tilt = abs(r['pitch_deg']) if r.get('pitch_deg') is not None else math.nan
        rr.RUN_BASE['name'] = base
        rr.VERIFY_MAX_READS = 1          # xy is reported, not judged: one marker read (Speed T1; up to 8 cost ~10 s)
        # ...and only for the mate being certified: a predecessor's marker xy is reported nowhere the verdict looks
        # (its depth decides), and each read is a verify_assembly process (~2.5 s; mate:03 paid four)
        is_last = (asm, addr, m) is R_[-1]
        v = rr._read_part(obj) if is_last else {'result': 'skipped', 'error': 'predecessor: marker xy not read (depth decides)'}
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
        te, why = tap_ends(obj, base)
        if te is not None and math.isfinite(z_err):
            ends = te                                  # contact, while nothing has moved the part since
            ev['ends_from'] = f'{why} -> ends {te} mm (depth read {ev["depth"].get("end_heights_mm")})'
        else:
            ev['ends_from'] = f'depth ({why})'
        if math.isfinite(z_err) and ends and min(ends) / 1000.0 < -lim_z:
            # the pocket floor stops the part: an end read below the seat by more than the band is the view, not
            # the part (2026-10-21: u_brown seen across the base read -2.29 mm at one end) -> not measured
            ev['invalid'] = f'an end {min(ends):.2f} mm below the seat - physically impossible, reading rejected'
            z_err = tilt = math.nan
        if not math.isfinite(z_err):
            inside, method = False, 'markers' if xy_seen else 'none'
            ev['verdict'] = 'not certified: ' + (ev.get('invalid') or 'seat height not measured from this view')
        else:
            # end heights are judged only where the mover can wedge looking seated (CAD); elsewhere they are reported:
            # a seated u_orange (flat by touch, ends +0.1 / -0.2 mm) read +0.89 / -1.79 and +2.38 / -0.16 mm from two
            # views a minute apart (10-22 11:50) - the centre height held within 1.6 mm
            from compliant_insertion_studio.wrapper.cad_geometry import wedge_possible
            from seams.acceptance import seat_inside          # the same rule the planner's seat recovery stops on
            ev['ends_judged'] = wedge_possible(base, obj)
            inside, ev['why'] = seat_inside(base, obj, z_err, ends, ends_by_contact=te is not None)
            method = 'both' if xy_seen else 'depth_seat_check'
            if not inside and not ev['ends_judged']:
                # a mover CAD says cannot rest off its seat inside the at-rest band (no wedge: out of the pocket it sits on
                # the rim, ~30 mm up) that the insert drove to seat depth IS seated - the depth read is the one off (operator
                # 10-24: "there is literally no way it can roll on any wedge"; u_brown read +2.92 / +3.01 / +2.74 mm after
                # inserts at +0.28 / +0.50 mm, then -0.91 / -0.75 mm untouched an hour later; nothing under it, not deformed)
                ins = _insert_depth(obj, lim_z)
                if ins is not None:
                    inside, method = True, 'insert_depth'
                    ev['why'] = f"depth read {ev['why']}; the insert seated it at {ins:+.2f} mm and CAD rules out a wedge"

            ev['verdict'] = ('inside' if inside else 'outside') + f" (xy implied by seat depth): {ev['why']}"
        out.append({'mate_id': f'{asm}/{addr}', 'inside': inside, 'error_mate_frame_m': [ex, ey, z_err],
                    'tilt_deg': tilt, 'method': method, 'evidence': ev})
        lap(f'mate {addr}')
    T.pop('_last', None); T['total'] = round(time.time() - T.pop('t0'), 2)
    print('[certify timing] ' + json.dumps(T), file=sys.stderr, flush=True)
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
