#!/usr/bin/env python3
"""drive-mate's align step (2026-10-22, AR3): bring the held part over its seat before the insert. Run by
drive_mate_server as a child process (the camera reads and the wrist turns call rclpy.init of their own; the server's
context is live), in the insert's process group, so a cancel reaches it the same way.

    python3 -m seams.drive_align --obj u_brown --base base1 --grasp-id 1 \\
        --measured X Y Z QX QY QZ QW --held-quat QX QY QZ QW
    -> __ALIGN__ {"aim_base": [[x, y, z], [qx, qy, qz, qw]], "measured_base": [...], "held_quat": [...],
                  "route": "...", "measured": bool}

Decision 10-22 (ARCHITECTURE_SEAMS.md): vision measures, drive-mate moves. The alignment is today's
(replay_real_assembly.align_for_insert: in-hand base re-read, then by route the marker servo / depth-frame servo /
yaw + close read), now run by drive-mate instead of the planner. Known debt, recorded here rather than hidden: it still
imports the replay module for those functions (the reverse dependency the next AR3 steps remove, moving them behind
this module's interface).
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'ablations')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--grasp-id', type=int, required=True)
    ap.add_argument('--measured', type=float, nargs=7, required=True)
    ap.add_argument('--held-quat', type=float, nargs=4, required=True)
    a = ap.parse_args()
    sys.argv = ['replay_real_assembly']
    import replay_real_assembly as rr
    rr.RUN_BASE['name'] = a.base
    # the replay's timing rows (every subprocess, camera read, servo step) for this align, beside its log (TGOAL:
    # run 32's aligns took 75-128 s and their logs carry no times)
    import time as _t
    rr._TIMING['fh'] = open(REPO / 'compliant_insertion_studio' / 'logs' /
                            f"drive_align_{a.obj}_{_t.strftime('%Y%m%d_%H%M%S')}.timing.jsonl", 'a', buffering=1)
    import compliant_insertion_studio.wrapper.marker_yaw as _my    # the servo's camera reads, timed (logging only)
    for _fn in ('still_frame', 'base_pose_same_frame', 'socket_shift_world', 'held_outline_px', 'base_pose_world'):
        setattr(_my, _fn, rr._timed(getattr(_my, _fn), 'camera', lambda a_, k_, n_=_fn: n_))
    # the rest of the align's still time (10-23 still audit: ~120 s per run inside drive-mate had no row)
    import base_icp as _bi
    _bi.read_icp = rr._timed(_bi.read_icp, 'camera', lambda a_, k_: 'read_icp (in-hand base fit)')
    import compliant_insertion_studio.wrapper.align_at_hover as _ah
    _ah.turn_wrist = rr._timed(_ah.turn_wrist, 'move', lambda a_, k_: f'turn_wrist {a_[0] if a_ else ""}')
    rr.snapshot_robot_state = rr._timed(rr.snapshot_robot_state, 'state', lambda a_, k_: 'snapshot_robot_state')
    os.environ.setdefault('SEAMS_DRIVE', '1')
    measured = (tuple(a.measured[:3]), tuple(a.measured[3:]))

    def say(m):
        print(m, flush=True)
    al = rr.align_for_insert(a.obj, a.base, measured, list(a.held_quat), a.grasp_id, say)

    def pose(p):
        return None if p is None else [list(map(float, p[0])), list(map(float, p[1]))]
    out = dict(aim_base=pose(al['aim_base']), measured_base=pose(al['measured_base']),
               held_quat=None if al['held_quat'] is None else list(map(float, al['held_quat'])),
               route=al['route'], measured=bool(rr.SERVO_MEASURED.get(a.obj)), safety=rr._SAFETY_STOP or '',
               grasp_depth_fix_m=rr.GRASP_DEPTH_FIX.get(a.obj))   # the servo's touchdown measured the grasp depth
    print('__ALIGN__ ' + json.dumps(out), flush=True)
    return 0 if not rr._SAFETY_STOP else 2


if __name__ == '__main__':
    sys.exit(main())
