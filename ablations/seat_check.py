#!/usr/bin/env python3
"""Post-insert seat check + minimal retry, measured from the post-release safe-height pose.

Why (2026-10-07): line_green can report SUCCESS from prismatic_peg_insertion while seated
wrong, and no insert telemetry separates it (QUEUED_FIXES #0). The camera can, from the
pose the sequence already reaches after release (move_to_safe_height straight above the
part): relative to base1, line_green read dy +8.5 / +12.3 mm when wrong and +2.1 when
correct, with +-0.1 mm frame noise (poses_line_green_{WRONG,WRONG2,CORRECT}.json).

Check: median xy error of the part vs its assembly target, over ~2.5 s, minus the part's
correct-seat reference (the camera has a per-part bias); bad if > tol.
Retry (minimal): regrasp in place -> lift 10 mm (not safe height) -> insert again (the
insert re-centres on the target seat; the regrasp centred the gripper on the part, so
this corrects the measured offset) -> release -> safe height -> re-check.

READ AT SAFE HEIGHT ONLY: the reference was recorded there; other viewpoints are 6-9x
noisier. Standalone: put the arm at the post-release pose first.

    python3 ablations/seat_check.py --object line_green            # check + retry
    python3 ablations/seat_check.py --object line_green --check-only
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Correct-seat reference (median error vs target, mm) and tolerance, per object.
# line_green: one correct + two wrong recordings, 2026-10-07 — widen with more data.
SEAT_REF = {
    'line_green': {'ref_mm': (1.53, 2.08), 'tol_mm': 4.0, 'grasp_id': 1},
}
LIFT_M = 0.010
MAX_RETRIES = 2


def _env():
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{env['PYTHONPATH']}" if 'PYTHONPATH' in env else str(REPO)
    return env


def _run(argv, timeout=180):
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=_env(),
                       cwd=str(REPO))
    out = p.stdout + p.stderr
    js = [l for l in out.splitlines() if l.startswith('{"result"')]
    res = json.loads(js[-1]) if js else {}
    return p.returncode == 0 and res.get('result') != 'failure', res, out


def measure(obj, secs=2.5):
    """(dx, dy, n): median xy error of `obj` vs its base1 target, in mm, or None."""
    ok, res, out = _run([sys.executable, __file__, '--measure-json', obj, str(secs)], 30)
    m = re.findall(r'^MEASURE (.*)$', out, re.M)
    if not m:
        return None
    d = json.loads(m[-1])
    return (d['dx'], d['dy'], d['n']) if d.get('n') else None


def _measure_main(obj, secs):
    import time
    import numpy as np
    import rclpy
    from scipy.spatial.transform import Rotation as R
    from tf2_msgs.msg import TFMessage
    from utils.data_path_finder import find_assembly_json_by_base_name, get_assembly_data_dir
    tgt = next(c for c in json.load(open(find_assembly_json_by_base_name(
        'base1', str(get_assembly_data_dir()))))['components'] if c['name'] == obj)['position']
    errs = []

    def cb(m):
        f = {t.child_frame_id: t.transform for t in m.transforms}
        if 'base1' in f and obj in f:
            b, o = f['base1'], f[obj]
            Rb = R.from_quat([b.rotation.x, b.rotation.y, b.rotation.z, b.rotation.w])
            rel = Rb.inv().apply([o.translation.x - b.translation.x,
                                  o.translation.y - b.translation.y,
                                  o.translation.z - b.translation.z])
            errs.append((1000 * (rel[0] - tgt['x']), 1000 * (rel[1] - tgt['y'])))
    rclpy.init()
    n = rclpy.create_node('seat_check_measure')
    n.create_subscription(TFMessage, '/objects_poses_real', cb, 10)
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.05)
    n.destroy_node()
    rclpy.shutdown()
    if not errs:
        print('MEASURE ' + json.dumps({'n': 0}))
        return
    a = np.array(errs)
    print('MEASURE ' + json.dumps({'dx': float(np.median(a[:, 0])), 'dy': float(np.median(a[:, 1])),
                                   'n': len(a)}))


def check(obj, say=print):
    ref = SEAT_REF[obj]
    m = measure(obj)
    if m is None:
        # Fail closed: the check is only valid from the post-release safe-height pose; not
        # seeing the part there is not evidence it is seated (2026-10-07).
        say(f"  [seat-check] {obj}: NOT OBSERVED — cannot judge -> treating as FAILED")
        return False, None
    dx, dy, n = m
    ex, ey = dx - ref['ref_mm'][0], dy - ref['ref_mm'][1]     # offset from a correct seat
    dist = math.hypot(ex, ey)
    ok = dist <= ref['tol_mm']
    say(f"  [seat-check] {obj}: err vs target ({dx:+.1f}, {dy:+.1f}) mm, n={n} -> "
        f"{dist:.1f} mm from correct-seat ref (tol {ref['tol_mm']:.0f}) -> {'OK' if ok else 'BAD SEAT'}")
    return ok, (ex, ey)


RETRY_ENABLED = False   # check-only unless explicitly allowed (--allow-retry); see below


def retry_once(obj, width_mm, offset_mm, say=print, log_dir=None):
    """Minimal retry, fail closed. Returns False (and stops) on the first failed step.

      regrasp in place -> lift LIFT_M straight up -> translate by -offset (the measured
      error from a correct seat; an estimate) at that height -> prismatic_peg_insertion
      DIRECTLY (no _run_hover: translate_object's line_green route always re-approaches
      from safe height) -> release -> safe height.

    All straight moves are move_to_safe_height --lift-by / --translate-xy (Cartesian waypoint
    IK seeded from the current joints, orientation kept). NOT primitives.core.move_xyz
    (fixed-seed IK, joint-space — swung the arm on 2026-10-07)."""
    if not RETRY_ENABLED:
        say("  [seat-check] retry not enabled (check-only) — not moving")
        return False
    import time as _t
    from primitives.shared.config import (DEFAULT_BASE_ORIENTATION, DEFAULT_BASE_POSITION,
                                          get_object_base_offset_m)
    log_dir = Path(log_dir or '/tmp/seat_check_logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _t.strftime('%H%M%S')
    gid = SEAT_REF[obj]['grasp_id']
    py = [sys.executable, '-u', '-m']

    def step(name, argv, timeout=180):
        ok, res, out = _run(argv, timeout)
        (log_dir / f"{stamp}_{name.replace(' ', '_')}.log").write_text(out)
        why = '' if ok else (res.get('error') or res.get('outcome_reason') or 'see log')
        say(f"  [seat-check]   {name}: {'ok' if ok else 'FAILED ' + str(why)}")
        return ok, res, out

    ok, res, _ = step('regrasp', py + ['primitives.move_to_grasp', '--object-name', obj,
                      '--grasp-id', str(gid), '--mode', 'real', '--base-name', 'base1',
                      '--pre-grasp-width', f"{width_mm:g}"])
    if not ok:
        return False
    q = (res.get('current_object_orientation') or {}).get('quat')
    if not step('close', py + ['primitives.control_gripper', 'close', '--mode', 'real'])[0]:
        return False
    if not step('lift', py + ['primitives.move_to_safe_height', '--mode', 'real',
                              '--lift-by', f"{LIFT_M:.3f}"])[0]:
        return False
    tx, ty = -offset_mm[0] / 1000.0, -offset_mm[1] / 1000.0     # move the part back by its error
    if max(abs(tx), abs(ty)) > 0.02:
        say(f"  [seat-check]   offset ({1000*tx:+.1f}, {1000*ty:+.1f}) mm exceeds 20 mm — stopping")
        return False
    if not step('translate', py + ['primitives.move_to_safe_height', '--mode', 'real',
                                   '--translate-xy', f"{tx:.4f}", f"{ty:.4f}"])[0]:
        return False
    off = get_object_base_offset_m(obj)
    base = [float(DEFAULT_BASE_POSITION[i]) + float(off[i]) for i in range(3)]
    argv = py + ['primitives._real_mode_stash.prismatic_peg_insertion', '--object-name', obj,
                 '--base-name', 'base1', '--grasp-id', str(gid), '--force', '2.0',
                 '--timeout', '60', '--final-base-pos', *[f"{v:.6f}" for v in base],
                 '--final-base-orientation', *[f"{v:.6f}" for v in DEFAULT_BASE_ORIENTATION]]
    if q:
        argv += ['--current-object-orientation', *[f"{q[k]:.6f}" for k in 'xyzw']]
    ok, _, out = step('insert (direct)', argv, 600)
    if not ok or '[SUCCESS]' not in out:
        say("  [seat-check]   insert did not seat — stopping with the part held, arm stationary")
        return False
    if not step('release', py + ['primitives.control_gripper', f"{width_mm:g}", '--mode', 'real'])[0]:
        return False
    return step('safe height', py + ['primitives.move_to_safe_height', '--mode', 'real'])[0]


def run(obj, width_mm, say=print, max_retries=MAX_RETRIES):
    """Check; on a bad seat retry minimally up to max_retries. True if it ends seated."""
    for attempt in range(max_retries + 1):
        ok, offset = check(obj, say)
        if ok:
            return True
        if offset is None or attempt == max_retries:
            say(f"  [seat-check] {obj} not confirmed seated after {attempt} retr{'y' if attempt == 1 else 'ies'}")
            return False
        say(f"  [seat-check] {obj}: minimal retry {attempt + 1}/{max_retries} (regrasp -> lift "
            f"{1000 * LIFT_M:.0f} mm -> translate ({-offset[0]:+.1f}, {-offset[1]:+.1f}) mm -> insert)")
        if not retry_once(obj, width_mm, offset, say):
            return False        # fail closed: never fall through to another check
    return False


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--measure-json':
        return _measure_main(sys.argv[2], float(sys.argv[3]))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--object', required=True, choices=sorted(SEAT_REF))
    ap.add_argument('--width-mm', type=float, default=None)
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--allow-retry', action='store_true',
                    help='Enable the minimal retry (regrasp, 10 mm lift, insert). Off by default.')
    a = ap.parse_args()
    global RETRY_ENABLED
    RETRY_ENABLED = bool(a.allow_retry)
    if a.check_only:
        sys.exit(0 if check(a.object)[0] else 1)
    width = a.width_mm
    if width is None:
        from primitives.shared.config import get_gripper_width_mm
        width = get_gripper_width_mm(a.object, SEAT_REF[a.object]['grasp_id'])
    sys.exit(0 if run(a.object, width) else 1)


if __name__ == '__main__':
    main()
