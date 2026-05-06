#!/usr/bin/env python3
"""
Regime segmentation for an insertion telemetry CSV.

See analysis/REGIME_DECODING.md sections 1-2 for the regime hypotheses and
detector definitions this implements.

Output: timeline of (t_start, t_end, regime, summary_stats) for one CSV.

Usage:
  python3 30_segment_regimes.py <basename>
  e.g.
  python3 30_segment_regimes.py insert_u_orange_20260504_113809
"""
import argparse
import csv
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _paths import LOG_DIR


# ---------------------------------------------------------------------------
# Threshold table (initial estimates; tunable from REGIME_DECODING.md §6)
# ---------------------------------------------------------------------------
TH = {
    'rim': dict(dz_dt_max=0.5e-3, fz_min=3.0, fz_max=8.0, tilt_max=1.5,
                T_lat_max=0.3),
    'edge': dict(dz_dt_min=0.5e-3, dz_dt_max=3.0e-3, fz_min=3.0,
                 tilt_min=1.5, T_lat_min=0.4),
    'chamfer': dict(dz_dt_min=3.0e-3, fz_min=2.0, tilt_swing_min=1.0),
    'descent': dict(dz_dt_min=1.0e-3, tilt_max=1.0, T_lat_max=0.2,
                    fz_max=5.0, tilt_swing_max=0.5),
    'seated': dict(dz_dt_max=0.3e-3, tilt_max=1.0,
                   z_to_seat_max=5e-3, sustain_s=1.0),
}
WINDOW_HALF_S = 0.25      # ± 0.25s = 0.5s window
DEBOUNCE_S    = 0.30
DT            = 0.01      # main CSV is ~100 Hz


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_active(basename):
    """Load ACTIVE-phase samples from the main CSV. Returns dict of np arrays."""
    csv_path = os.path.join(str(LOG_DIR), f"{basename}.csv")
    rows = list(csv.DictReader(open(csv_path)))
    active = [r for r in rows if r.get('phase', '') == 'ACTIVE']
    if not active:
        raise SystemExit(f"No ACTIVE rows in {csv_path}")

    def f(r, k):
        try: return float(r[k])
        except: return float('nan')

    cols = ['t_s', 'tcp_x', 'tcp_y', 'tcp_z',
            'tcp_qx', 'tcp_qy', 'tcp_qz', 'tcp_qw',
            'fx', 'fy', 'fz', 'tx', 'ty', 'tz']
    return {c: np.array([f(r, c) for r in active]) for c in cols}


def load_predicted_seat_z(basename):
    """Pull predicted_tcp_at_seat.z from the meta file (CAD chain output)."""
    meta_path = os.path.join(str(LOG_DIR), f"{basename}.meta.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path))
    cad = meta.get('cad_prediction', {})
    pt  = cad.get('predicted_tcp_at_seat', {}).get('xyz_m')
    return pt[2] if pt and len(pt) >= 3 else None


# ---------------------------------------------------------------------------
# Derived signals
# ---------------------------------------------------------------------------
def tilt_deg_series(qx, qy, qz, qw):
    """Angle between EE +Z axis and world -Z axis, in degrees."""
    from scipy.spatial.transform import Rotation as R
    out = np.empty(len(qx))
    for i in range(len(qx)):
        r = R.from_quat([qx[i], qy[i], qz[i], qw[i]])
        ee_z = r.apply([0, 0, 1])
        out[i] = np.degrees(np.arccos(np.clip(-ee_z[2], -1, 1)))
    return out


def tilt_components_series(qx, qy, qz, qw):
    """Decompose tilt into directional components.

    Returns (tilt_x_deg, tilt_y_deg, yaw_drift_deg) where:
      - tilt_x: peg leans in +X world (positive) or -X (negative)
      - tilt_y: peg leans in +Y world (positive) or -Y (negative)
      - yaw_drift: rotation around world -Z axis from canonical face-down

    The xy components localize WHICH direction peg is being pinned at the rim.
    The contact reaction tells you "rim is on the side opposite the tilt."
    """
    from scipy.spatial.transform import Rotation as R
    n = len(qx)
    tx = np.empty(n)
    ty = np.empty(n)
    yaw = np.empty(n)
    for i in range(n):
        r = R.from_quat([qx[i], qy[i], qz[i], qw[i]])
        ee_z = r.apply([0, 0, 1])  # peg axis in world (peg points down → ee_z ≈ (0,0,-1))
        # Tilt vector in world xy = projection of ee_z onto xy plane (after sign flip
        # since face-down ee_z is -world_z, lean direction = -ee_z[xy])
        tx[i] = np.degrees(np.arctan2(-ee_z[0], -ee_z[2]))
        ty[i] = np.degrees(np.arctan2(-ee_z[1], -ee_z[2]))
        # Yaw drift: rotation about world Z that the peg has from face-down canonical
        # Face-down canonical EE = R.from_euler('xyz', [180, 0, 180]) — yaw around Z = 180
        # Compute current yaw via euler decomposition
        rpy = r.as_euler('xyz', degrees=True)
        # Normalize yaw to [-180, 180] difference from canonical 180
        canonical_yaw = 180.0
        delta = (rpy[2] - canonical_yaw + 540) % 360 - 180
        yaw[i] = delta
    return tx, ty, yaw


def derive_signals(a):
    """Add derived columns: dz_dt (smoothed), T_lat magnitude, tilt_deg."""
    fs = round(1.0 / DT)
    smooth = lambda x, n=fs // 5: np.convolve(x, np.ones(n) / n, mode='same')
    a['fz_smooth'] = smooth(np.abs(a['fz']))
    a['T_lat'] = np.hypot(a['tx'], a['ty'])
    a['F_lat'] = np.hypot(a['fx'], a['fy'])
    a['tilt_deg'] = tilt_deg_series(a['tcp_qx'], a['tcp_qy'], a['tcp_qz'], a['tcp_qw'])
    tx, ty, yaw = tilt_components_series(a['tcp_qx'], a['tcp_qy'], a['tcp_qz'], a['tcp_qw'])
    a['tilt_x_deg'] = tx
    a['tilt_y_deg'] = ty
    a['yaw_drift_deg'] = yaw
    # dz/dt over 0.2s window (4 samples)
    dz = np.zeros_like(a['tcp_z'])
    win = max(1, int(0.2 * fs))
    for i in range(len(dz)):
        a0, a1 = max(0, i - win), min(len(dz) - 1, i + win)
        if a1 - a0 > 0:
            dz[i] = (a['tcp_z'][a1] - a['tcp_z'][a0]) / ((a1 - a0) * DT)
    a['dz_dt'] = dz   # sign: + means descending (tcp_z increases) — NOTE ROS Z convention
    return a


# ---------------------------------------------------------------------------
# Window helper
# ---------------------------------------------------------------------------
def windowed(a, idx, half_s=WINDOW_HALF_S):
    """Return medians + min/max for a window centered at idx."""
    fs = round(1.0 / DT)
    half = max(1, int(half_s * fs))
    s = slice(max(0, idx - half), min(len(a['t_s']), idx + half))
    return {
        'fz':       float(np.median(a['fz_smooth'][s])),
        'T_lat':    float(np.median(a['T_lat'][s])),
        'tilt':     float(np.median(a['tilt_deg'][s])),
        'tilt_max': float(np.max(a['tilt_deg'][s])),
        'tilt_min': float(np.min(a['tilt_deg'][s])),
        'dz_dt':    float(np.median(a['dz_dt'][s])),
        'F_lat':    float(np.median(a['F_lat'][s])),
    }


# ---------------------------------------------------------------------------
# Detectors (Section 2.1 of REGIME_DECODING.md)
# ---------------------------------------------------------------------------
def is_RIM(w):
    th = TH['rim']
    return (
        abs(w['dz_dt']) < th['dz_dt_max']
        and th['fz_min'] < w['fz'] < th['fz_max']
        and w['tilt'] < th['tilt_max']
        and w['T_lat'] < th['T_lat_max']
    )


def is_EDGE(w):
    th = TH['edge']
    if abs(w['dz_dt']) < th['dz_dt_min'] or abs(w['dz_dt']) >= th['dz_dt_max']:
        return False
    if w['fz'] < th['fz_min']:
        return False
    return w['tilt'] > th['tilt_min'] or w['T_lat'] > th['T_lat_min']


def is_CHAMFER(w):
    th = TH['chamfer']
    return (
        abs(w['dz_dt']) >= th['dz_dt_min']
        and (w['tilt_max'] - w['tilt_min']) > th['tilt_swing_min']
        and w['fz'] > th['fz_min']
    )


def is_DESCENT(w):
    th = TH['descent']
    return (
        abs(w['dz_dt']) >= th['dz_dt_min']
        and w['tilt'] < th['tilt_max']
        and w['T_lat'] < th['T_lat_max']
        and w['fz'] < th['fz_max']
        and (w['tilt_max'] - w['tilt_min']) < th['tilt_swing_max']
    )


def is_SEATED(w, tcp_z, predicted_z, in_state_s):
    th = TH['seated']
    if predicted_z is None:
        return False
    return (
        abs(w['dz_dt']) < th['dz_dt_max']
        and abs(tcp_z - predicted_z) < th['z_to_seat_max']
        and w['tilt'] < th['tilt_max']
        and in_state_s >= th['sustain_s']
    )


def classify(w, tcp_z, predicted_z, in_state_s):
    """Return regime label in priority order (most specific first)."""
    if is_SEATED(w, tcp_z, predicted_z, in_state_s):
        return 'SEATED'
    if is_CHAMFER(w):
        return 'CHAMFER_TRANSIT'
    if is_EDGE(w):
        return 'EDGE_OF_SLOT'
    if is_DESCENT(w):
        return 'IN_SLOT_DESCENT'
    if is_RIM(w):
        return 'RIM'
    return 'UNKNOWN'


# ---------------------------------------------------------------------------
# Segmentation w/ debounce
# ---------------------------------------------------------------------------
def find_first_contact(a, fz_threshold=5.0, sustain_s=0.1):
    """Index of first sample where smoothed |fz| > threshold for sustain_s."""
    fs = round(1.0 / DT)
    n_sustain = max(1, int(sustain_s * fs))
    above = a['fz_smooth'] > fz_threshold
    for i in range(len(above) - n_sustain):
        if above[i:i + n_sustain].all():
            return i
    return None


def segment(a, predicted_z=None, debounce_s=DEBOUNCE_S):
    fs = round(1.0 / DT)
    debounce_n = max(1, int(debounce_s * fs))
    n = len(a['t_s'])
    timeline = []
    candidate, candidate_start, candidate_count = None, 0, 0
    committed, committed_start = None, 0

    contact_i = find_first_contact(a)
    # Pre-contact span: emit a single APPROACH segment if there's any
    if contact_i is not None and contact_i > 0:
        timeline.append({
            'regime': 'APPROACH', 'start_s': float(a['t_s'][0]),
            'end_s': float(a['t_s'][contact_i]),
            'duration_s': float(a['t_s'][contact_i] - a['t_s'][0]),
            'i_start': 0, 'i_end': contact_i,
        })
        start_i = contact_i
    else:
        start_i = 0

    for i in range(start_i, n):
        w = windowed(a, i)
        in_state_s = (i - committed_start) * DT if committed else 0
        regime = classify(w, a['tcp_z'][i], predicted_z, in_state_s)

        if regime == candidate:
            candidate_count += 1
        else:
            candidate = regime
            candidate_start = i
            candidate_count = 1

        if candidate_count >= debounce_n and candidate != committed:
            if committed is not None:
                t0 = a['t_s'][committed_start]
                t1 = a['t_s'][candidate_start]
                timeline.append({
                    'regime': committed,
                    'start_s': float(t0),
                    'end_s':   float(t1),
                    'duration_s': float(t1 - t0),
                    'i_start': committed_start,
                    'i_end':   candidate_start,
                })
            committed, committed_start = candidate, candidate_start

    if committed is not None:
        t0 = a['t_s'][committed_start]
        t1 = a['t_s'][n - 1]
        timeline.append({
            'regime': committed,
            'start_s': float(t0),
            'end_s':   float(t1),
            'duration_s': float(t1 - t0),
            'i_start': committed_start,
            'i_end':   n - 1,
        })

    return timeline


def summarize_segment(a, seg):
    """Per-segment summary stats (used by the operator-direction decoder downstream)."""
    s = slice(seg['i_start'], seg['i_end'] + 1)
    if seg['i_end'] - seg['i_start'] < 2:
        return {**seg}
    dx = a['tcp_x'][seg['i_end']] - a['tcp_x'][seg['i_start']]
    dy = a['tcp_y'][seg['i_end']] - a['tcp_y'][seg['i_start']]
    dz = a['tcp_z'][seg['i_end']] - a['tcp_z'][seg['i_start']]
    travel_xy_mm = np.hypot(dx, dy) * 1000
    return {
        **seg,
        'tcp_dxy_mm': float(travel_xy_mm),
        'tcp_dz_mm':  float(dz * 1000),
        'fx_med': float(np.median(a['fx'][s])),
        'fy_med': float(np.median(a['fy'][s])),
        'fz_med': float(np.median(a['fz'][s])),
        'T_lat_med': float(np.median(a['T_lat'][s])),
        'tilt_med': float(np.median(a['tilt_deg'][s])),
        'tilt_peak': float(np.max(a['tilt_deg'][s])),
        # Directional tilt components — which-side-is-pinned signal
        'tilt_x_med':   float(np.median(a['tilt_x_deg'][s])),
        'tilt_y_med':   float(np.median(a['tilt_y_deg'][s])),
        'tilt_x_range': float(np.max(a['tilt_x_deg'][s]) - np.min(a['tilt_x_deg'][s])),
        'tilt_y_range': float(np.max(a['tilt_y_deg'][s]) - np.min(a['tilt_y_deg'][s])),
        'yaw_drift_med': float(np.median(a['yaw_drift_deg'][s])),
        'yaw_drift_range': float(np.max(a['yaw_drift_deg'][s]) - np.min(a['yaw_drift_deg'][s])),
        'dx_unit_xy': float(dx / (travel_xy_mm * 1e-3)) if travel_xy_mm > 0.5 else None,
        'dy_unit_xy': float(dy / (travel_xy_mm * 1e-3)) if travel_xy_mm > 0.5 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('basename')
    p.add_argument('--out', default=None,
                   help='write JSON timeline to this path (default: stdout pretty-print)')
    args = p.parse_args()

    a = load_active(args.basename)
    a = derive_signals(a)
    pz = load_predicted_seat_z(args.basename)
    timeline = segment(a, predicted_z=pz)
    timeline = [summarize_segment(a, s) for s in timeline]

    print(f"\n=== {args.basename}  ({len(timeline)} segments)  predicted_seat_z={pz} ===")
    print(f"  {'regime':<18s}  {'start_s':>8s}  {'dur_s':>6s}  "
          f"{'dxy_mm':>7s}  {'dz_mm':>7s}  {'fz_med':>7s}  "
          f"{'T_lat':>6s}  {'tilt_p':>6s}  {'unit_xy':>14s}")
    for s in timeline:
        ux = f"{s.get('dx_unit_xy'):+.2f}" if s.get('dx_unit_xy') is not None else '   - '
        uy = f"{s.get('dy_unit_xy'):+.2f}" if s.get('dy_unit_xy') is not None else '   - '
        print(f"  {s['regime']:<18s}  {s['start_s']:>8.2f}  {s['duration_s']:>6.2f}  "
              f"{s.get('tcp_dxy_mm',0):>7.2f}  {s.get('tcp_dz_mm',0):>+7.2f}  "
              f"{s.get('fz_med',0):>+7.2f}  {s.get('T_lat_med',0):>6.3f}  "
              f"{s.get('tilt_peak',0):>6.2f}  ({ux},{uy})")

    if args.out:
        with open(args.out, 'w') as fh:
            json.dump({'basename': args.basename, 'timeline': timeline,
                       'predicted_seat_z': pz, 'thresholds': TH}, fh, indent=2)
        print(f"\nwrote: {args.out}")


if __name__ == '__main__':
    main()
