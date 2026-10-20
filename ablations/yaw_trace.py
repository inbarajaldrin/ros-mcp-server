#!/usr/bin/env python3
"""Where does a part's entry yaw error come from: the pick, or the move to the insert?

Per insert (replay log + its robot-state snapshots + the insert meta), all from measurements, none from the
chain's own held-orientation bookkeeping:

  part@pick   the camera's orientation of the part on the table (move_to_grasp's current_object_orientation)
  wrist@pick  the TCP at the grasp (snapshot after move_to_grasp)
  grasp yaw   wrist@pick vs part@pick about world z (modulo the part's symmetry) - what move_to_grasp aligned
  wrist@seat  the TCP after the insert (snapshot after translate_object; rotation is locked in the search)
  entry err   the PHYSICAL part at the seat (wrist@seat * wrist@pick^-1 * part@pick, a rigid hold assumed)
              vs its seat target (camera-read base * CAD seat), yaw about world z, modulo symmetry
  Tz          median twisting torque during the search (insert_diagnose)
  seated      the camera's yaw error of the seated part after release ([verify] line)

If entry err tracks Tz, the yaw comes from the pick/hold chain; if entry err is ~0 but Tz is not, the camera's
part@pick (or the base read) is what is off.

  python3 ablations/yaw_trace.py [--part u_brown] [--since 2026-10-20]
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'ablations'))
LOGS = REPO / 'ablations/logs'
ILOGS = REPO / 'compliant_insertion_studio/logs'


def yaw_between(Ra, Rb, part):
    """Yaw (deg, about world z) of Ra relative to Rb, minimised over Rb's symmetry-equivalent orientations."""
    from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
    from utils.data_path_finder import get_symmetry_dir
    try:
        fold = load_symmetry_data(part, str(get_symmetry_dir()))
        cands = [R.from_matrix(M) for M in equivalent_orientations(Rb.as_matrix(), fold)]
    except Exception:
        cands = [Rb]
    best = min(cands, key=lambda c: (Ra * c.inv()).magnitude())
    d = Ra * best.inv()
    return float(np.degrees(d.as_rotvec()[2])), float(np.degrees(np.hypot(*d.as_rotvec()[:2])))


def trace(log_path):
    part = re.search(r'replay_(u_brown|u_orange|line_green|inverted_u_yellow)_', log_path.name)
    if not part:
        return []
    part = part.group(1)
    snap_path = log_path.with_suffix('.snapshots.jsonl')
    if not snap_path.exists():
        return []
    snaps = [json.loads(l) for l in open(snap_path)]
    text = open(log_path).read()
    # camera orientation of the part at each move_to_grasp (in order)
    picks = [json.loads(m) for m in re.findall(r'(\{"result": "success", "object_name": "%s", "grasp_id": \d+, "mode": "real", "movement_type": "move_to_object".*?\})\n' % part, text)]
    verifies = re.findall(r'\[verify\] %s: OK \(marker xy ([0-9.]+) mm, yaw err ([-0-9.]+) deg\)' % part, text)
    out = []
    grasp_snaps = [s for s in snaps if s['when'] == 'after' and 'move_to_grasp' in s['label']]
    seat_snaps = [s for s in snaps if s['when'] == 'after' and 'translate_object' in s['label']]
    for k, ss in enumerate(seat_snaps):
        # the last grasp before this insert
        t_seat = dt.datetime.fromisoformat(ss['ts_iso'])
        gi = [i for i, g in enumerate(grasp_snaps) if dt.datetime.fromisoformat(g['ts_iso']) < t_seat]
        if not gi or len(picks) <= gi[-1]:
            continue
        g = grasp_snaps[gi[-1]]; pk = picks[gi[-1]]
        q = pk.get('current_object_orientation', {}).get('quat')
        if not q:
            continue
        R_part_pick = R.from_quat([q['x'], q['y'], q['z'], q['w']])
        R_w_pick = R.from_quat(g['snapshot']['tcp']['quat_xyzw'])
        R_w_seat = R.from_quat(ss['snapshot']['tcp']['quat_xyzw'])
        R_part_seat = R_w_seat * R_w_pick.inv() * R_part_pick        # rigid hold
        # the insert meta closest in time (its end ~ the snapshot after translate_object)
        metas = sorted(ILOGS.glob(f'insert_{part}_*.meta.json'),
                       key=lambda p: abs((dt.datetime.fromtimestamp(p.stat().st_mtime) - t_seat).total_seconds()))
        if not metas:
            continue
        mp = metas[0]
        if abs((dt.datetime.fromtimestamp(mp.stat().st_mtime) - t_seat).total_seconds()) > 120:
            continue
        m = json.load(open(mp))
        cp = m.get('cad_prediction') or {}
        if 'T_world_object_seat' not in cp:
            continue
        R_seat = R.from_quat(cp['T_world_object_seat']['quat_xyzw'])
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        _, sq = load_seat_pose_in_base('base1', part)
        base_yaw = float(np.degrees((R_seat * R.from_quat(sq).inv()).as_euler('xyz')[2]))
        entry, tilt = yaw_between(R_part_seat, R_seat, part)
        grasp, _ = yaw_between(R_w_pick, R_part_pick, part)
        try:
            from insert_diagnose import diagnose
            d = diagnose(mp) or {}
        except Exception:
            d = {}
        sv = verifies[min(k, len(verifies) - 1)] if verifies else None
        out.append(dict(run=mp.name.replace('insert_', '').replace('.meta.json', ''), base=base_yaw,
                        wrist_pick=float(np.degrees((R.from_quat([0, 1, 0, 0]).inv() * R_w_pick).as_euler('xyz')[2])),
                        part_pick=float(R_part_pick.as_euler('xyz', degrees=True)[2]), grasp=grasp,
                        wrist_seat=float(np.degrees((R.from_quat([0, 1, 0, 0]).inv() * R_w_seat).as_euler('xyz')[2])),
                        entry=entry, tilt=tilt, tz=d.get('tz', float('nan')), rim=d.get('rim_s', float('nan')),
                        seated=float(sv[1]) if sv else float('nan')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='u_brown')
    ap.add_argument('--since', default='2026-10-19')
    a = ap.parse_args()
    t0 = dt.datetime.fromisoformat(a.since).timestamp()
    rows = []
    for lp in sorted(LOGS.glob(f'replay_{a.part}_*.log'), key=lambda p: p.stat().st_mtime):
        if lp.stat().st_mtime < t0:
            continue
        try:
            rows += trace(lp)
        except Exception as e:
            print(f"  [skip] {lp.name}: {e}")
    print(f"{'insert':30s} {'base':>6s} {'grasp':>6s} {'entry':>6s} {'tilt':>5s} {'Tz':>7s} {'rim s':>6s} {'seated':>6s}")
    for r in rows:
        print(f"{r['run']:30s} {r['base']:+6.2f} {r['grasp']:+6.2f} {r['entry']:+6.2f} {r['tilt']:5.1f} {r['tz']:+7.3f} {r['rim']:6.1f} {r['seated']:+6.2f}")
    if len(rows) >= 3:
        e = np.array([r['entry'] for r in rows]); tz = np.array([r['tz'] for r in rows]); b = np.array([r['base'] for r in rows])
        ok = np.isfinite(tz)
        if ok.sum() >= 3:
            print(f"\ncorr(entry yaw, Tz) = {np.corrcoef(e[ok], tz[ok])[0,1]:+.2f}   corr(base yaw, Tz) = {np.corrcoef(b[ok], tz[ok])[0,1]:+.2f}   "
                  f"corr(base yaw, entry) = {np.corrcoef(b, e)[0,1]:+.2f}")


if __name__ == '__main__':
    main()
