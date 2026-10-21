#!/usr/bin/env python3
"""drive-mate on ROS (2026-10-20, M3a): fmb_seam_msgs/action/DriveMate at /ur5e/drive_mate.

    python3 -m seams.drive_mate_server

The insertion driver behind the decided seam (ARCHITECTURE_SEAMS.md): bring the held part into the mate's accepted
seated set under uncertainty and report what happened. Classical now (today's insert stack, unchanged); the RL
driver plugs in behind the same action later (M3b).

  goal    mate_id   fmb1/mate:00 - resolved through seams/bindings.yaml to the mover and its reference
          target    VisionPose of the REFERENCE (the base) in the robot base frame - what to aim at; any aim
                    corrections (seated-neighbour follow, seat-history bias) are the planner's and already applied
          held      VisionPose of the mover in the FLANGE frame (tool0) - the grasp. The grasp id and the grasp
                    depth are read off it: the CAD grasp point that lies on the tool axis (<= HELD_AXIS_TOL_M off
                    it) is the grasp, its tool-z coordinate the depth (GRASP_TOOL_OFFSET_M of the insert)
  result  seated          never the FSM label: where the arm RESTS after the insert (TCP still < 0.5 mm over 0.5 s)
                          with |tcp_z - predicted seat z| < SEATED_DZ_M. The trace alone cannot say it (its seat
                          detector ends logging on arrival, still descending); its own reading is kept as trace_stage
          stage_reached   seat | wedged (still, 0 < dz <= 7 mm, within 5 mm of the aim: on a channel edge) | above |
                          below | unsettled (the trace ends still moving) | no_trace.  Checked 2026-10-20 against the
                          last 40 FMB1 inserts: 38 agree with the FSM label, 2 ended within 1 mm but moving
          slide           end TCP xy - predicted seat xy (what the search had to correct)
          measured_correction  [dx, dy, dz] of the end TCP vs the prediction
          trace_path      the insert's raw CSV
          refusal         mate_unknown | assembly_unbound | assembly_ambiguous | target_frame | held_frame |
                          held_unmatched | insert_failed_to_start
  cancel  SIGINT to the insert (its force-mode cleanup runs), SIGTERM after 30 s - never SIGKILL in force mode.

Known debt, recorded here rather than hidden: the insert routes by part name (translate_object: lines -> prismatic,
U parts -> compliant spiral) and adds a per-part seat-z offset (PER_OBJECT_BASE_OFFSET_M) - the seam rule says
stages/routes come from CAD. Both live inside the primitive this server wraps; the next step is deriving the route
from the CAD stages (insert_stages.py) instead.
"""
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from fmb_seam_msgs.action import DriveMate

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
ACTION = '/ur5e/drive_mate'
B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))
LOGS = REPO / 'compliant_insertion_studio' / 'logs'
HELD_AXIS_TOL_M = 0.006          # a CAD grasp point this close to the tool axis is the grasp
SEATED_DZ_M = 0.005              # CLAUDE.md: |tcp_z - predicted_seat| < 5 mm and motion stopped
WEDGE_DZ_M, WEDGE_DXY_M = 0.007, 0.005
REST_WINDOW_S, REST_STILL_M = 0.5, 0.0005    # the arm at rest after the insert: < 0.5 mm over 0.5 s


def resolve(mate):
    asm, _, addr = mate.rpartition('/')
    bound = [k for k in B if k != 'sigma']
    if not asm:
        if len(bound) != 1:
            return None, 'assembly_ambiguous'
        asm = bound[0]
    if asm not in bound:
        return None, 'assembly_unbound'
    m = B[asm]['mates'].get(addr)
    if m is None:
        return None, 'mate_unknown'
    return (m['mover'], B[asm]['reference']), ''


def _grasp_points(obj):
    from primitives.shared.config import _GRASP_POINTS_DIR
    d = json.load(open(f'{_GRASP_POINTS_DIR}/{obj}_grasp_points.json'))
    return [(int(g['id']), np.array([g['position'][k] for k in 'xyz'], float)) for g in d['grasp_points']]


def held_pose(obj, grasp_id, held_quat_world, tcp_quat_world, grasp_depth_m):
    """For planners: the mover in the flange frame from today's chain (world orientation of the held part, its grasp
    id, the grasp depth move_to_grasp executed). Returns (xyz, quat_xyzw)."""
    Rfo = R.from_quat(tcp_quat_world).inv() * R.from_quat(held_quat_world)
    g = dict(_grasp_points(obj))[int(grasp_id)]
    p = np.array([0.0, 0.0, grasp_depth_m]) - Rfo.apply(g)
    return [float(v) for v in p], [float(v) for v in Rfo.as_quat()]


def grasp_from_held(obj, p_fo, q_fo):
    """(grasp_id, grasp_depth_m, off_axis_m) of the CAD grasp point nearest the tool axis, or None."""
    Rfo = R.from_quat(q_fo)
    best = None
    for gid, g in _grasp_points(obj):
        f = Rfo.apply(g) + p_fo                       # grasp point in the flange frame
        off = float(np.hypot(f[0], f[1]))
        if best is None or off < best[2]:
            best = (gid, float(f[2]), off)
    return best if best and best[2] <= HELD_AXIS_TOL_M else (None, None, best[2] if best else None)


def trace_outcome(obj, since):
    """Seat judged from the newest insert CSV written since `since` (the meta's predicted seat vs the end TCP)."""
    import csv
    metas = [m for m in glob.glob(str(LOGS / f'insert_{obj}_*.meta.json')) if os.path.getmtime(m) >= since - 1]
    if not metas:
        return {'stage_reached': 'no_trace'}
    return outcome_from_meta(max(metas, key=os.path.getmtime))


def outcome_from_meta(mp):
    import csv
    m = json.load(open(mp)); csv_path = mp.replace('.meta.json', '.csv')
    rows = [x for x in csv.DictReader(open(csv_path)) if x.get('tcp_z') not in ('', 'nan', None)]
    # the prismatic stash writes predicted_seat_xyz; the compliant FSM writes cad_prediction.predicted_tcp_at_seat
    seat = m.get('predicted_seat_xyz') or ((m.get('cad_prediction') or {}).get('predicted_tcp_at_seat') or {}).get('xyz_m')
    if not rows or not seat:
        return {'stage_reached': 'no_trace', 'trace_path': csv_path}
    P = np.array([[float(x['tcp_x']), float(x['tcp_y']), float(x['tcp_z'])] for x in rows[-50:]])
    tgt = np.array(seat, float)
    end = P[-1]; d = end - tgt; still = float(np.ptp(P[:, 2])) < 0.001
    dxy = float(np.hypot(d[0], d[1]))
    if still and abs(d[2]) < SEATED_DZ_M:
        stage = 'seat'
    elif still and 0 < d[2] <= WEDGE_DZ_M and dxy <= WEDGE_DXY_M:
        stage = 'wedged'
    elif not still:
        stage = 'unsettled'                                    # trace ends still moving: no judgement
    else:
        stage = 'above' if d[2] > 0 else 'below'
    return {'stage_reached': stage, 'seated': stage == 'seat', 'still': still, 'trace_path': csv_path,
            'seat_xyz': [float(v) for v in tgt],
            'slide': [float(d[0]), float(d[1])], 'measured_correction': [float(v) for v in d],
            'fsm_outcome': m.get('outcome'), 'router': m.get('router')}


def judge_at_rest(out, rest):
    """Seat judged where the arm came to rest after the insert (the decider); the trace's own reading is kept as
    trace_stage. slide / measured_correction are the resting TCP vs the predicted seat."""
    P, still = rest
    tgt = np.array(out['seat_xyz']); d = np.asarray(P) - tgt; dxy = float(np.hypot(d[0], d[1]))
    if not still:
        stage = 'unsettled'
    elif abs(d[2]) < SEATED_DZ_M:
        stage = 'seat'
    elif 0 < d[2] <= WEDGE_DZ_M and dxy <= WEDGE_DXY_M:
        stage = 'wedged'
    else:
        stage = 'above' if d[2] > 0 else 'below'
    return dict(out, trace_stage=out['stage_reached'], stage_reached=stage, seated=stage == 'seat', still=still,
                judged_on='resting TCP', slide=[float(d[0]), float(d[1])], measured_correction=[float(v) for v in d])


class DriveMateServer(Node):
    def __init__(self):
        super().__init__('ur5e_drive_mate')
        self.tcp = None; self.proc = None; self.busy = threading.Lock()
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: setattr(self, 'tcp', m.pose), 10)
        ActionServer(self, DriveMate, ACTION, self.execute, goal_callback=self.on_goal,
                     cancel_callback=self.on_cancel, callback_group=ReentrantCallbackGroup())
        self.get_logger().info(f'drive-mate served at {ACTION}')

    def resting_tcp(self, window_s=REST_WINDOW_S, timeout_s=5.0):
        """The TCP once the arm has stopped after the insert: (xyz, still) from a window of samples. The insert's
        trace ends when its seat detector fires - on arrival, still descending (2026-10-21 u_brown: trace ended
        moving at +1.5 mm, the arm rested at +1.3 mm, seated) - so the trace cannot say whether it came to rest."""
        t_end = time.time() + timeout_s; buf = []
        while time.time() < t_end:
            p = self.tcp
            if p is not None:
                buf.append((time.time(), p.position.x, p.position.y, p.position.z))
                w = [b for b in buf if b[0] >= buf[-1][0] - window_s]
                if buf[-1][0] - buf[0][0] >= window_s:
                    P = np.array([b[1:] for b in w])
                    if float(np.ptp(P, axis=0).max()) < REST_STILL_M:
                        return P[-1], True
            time.sleep(0.02)
        return (np.array(buf[-1][1:]), False) if buf else None

    def on_goal(self, goal):
        return GoalResponse.REJECT if self.busy.locked() else GoalResponse.ACCEPT

    def on_cancel(self, gh):
        p = self.proc
        if p is not None and p.poll() is None:
            p.send_signal(signal.SIGINT)                       # the insert's own force-mode cleanup
            threading.Thread(target=self._term_later, args=(p,), daemon=True).start()
        return CancelResponse.ACCEPT

    @staticmethod
    def _term_later(p):
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.terminate()                                      # never SIGKILL in force mode

    def execute(self, gh):
        with self.busy:
            return self._execute(gh)

    def _execute(self, gh):
        g = gh.request; res = DriveMate.Result()
        res.seated = False; res.stage_reached = ''; res.slide = [float('nan')] * 2
        res.measured_correction = [float('nan')] * 3; res.trace_path = ''

        def refuse(code, detail=''):
            res.refusal = code
            self.get_logger().warn(f'{g.mate_id}: refused {code} {detail}')
            gh.abort(); return res

        ok, code = resolve(g.mate_id)
        if not ok:
            return refuse(code)
        obj, base = ok
        if g.target.header.frame_id != 'base':
            return refuse('target_frame', g.target.header.frame_id)
        if g.held.header.frame_id != 'tool0':
            return refuse('held_frame', g.held.header.frame_id)
        if self.tcp is None:
            return refuse('insert_failed_to_start', 'no TCP pose')
        hp = g.held.pose
        p_fo = np.array([hp.position.x, hp.position.y, hp.position.z])
        q_fo = [hp.orientation.x, hp.orientation.y, hp.orientation.z, hp.orientation.w]
        gid, depth, off = grasp_from_held(obj, p_fo, q_fo)
        if gid is None:
            return refuse('held_unmatched', f'nearest CAD grasp point {1000 * (off or float("nan")):.1f} mm off the tool axis')
        tq = self.tcp.orientation
        held_world = (R.from_quat([tq.x, tq.y, tq.z, tq.w]) * R.from_quat(q_fo)).as_quat()
        tp = g.target.pose
        # read fresh every goal: a server started before a config edit kept the old offset and aimed yellow 6 mm
        # deep while the planner (a new process) used the new one (2026-10-21)
        import importlib
        import primitives.shared.config as _cfg
        importlib.reload(_cfg)
        dz = float(_cfg.get_object_base_offset_m(obj)[2])     # known debt: per-part seat-z offset (docstring)
        argv = ['python3', '-u', '-m', 'primitives.translate_object', '--mode', 'real', '--object-name', obj,
                '--base-name', base, '--insert', '--insertion-type', 'compliant', '--grasp-id', str(gid),
                '--current-object-orientation', *[f'{v:.6f}' for v in held_world],
                '--final-base-pos', f'{tp.position.x:.6f}', f'{tp.position.y:.6f}', f'{tp.position.z + dz:.6f}',
                '--final-base-orientation', *[f'{v:.6f}' for v in (tp.orientation.x, tp.orientation.y,
                                                                    tp.orientation.z, tp.orientation.w)]]
        env = os.environ.copy()
        env['GRASP_TOOL_OFFSET_M'] = f'{depth:.6f}'; env['ROS_MCP_MODE'] = 'real'
        # one attempt per goal, a wedged bar held where it stopped: retries and recoveries are the caller's (the
        # replay sets the same two at import)
        env['PRISMATIC_MAX_RETRIES'] = '1'; env['PRISMATIC_HOLD_IF_WEDGED'] = '1'
        env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH', '')}"
        self.get_logger().info(f'{g.mate_id}: {obj} into {base}, grasp {gid} at depth {1000 * depth:.1f} mm '
                               f'({1000 * off:.1f} mm off axis)')
        t0 = time.time()
        log = open(LOGS / f'drive_mate_{obj}_{time.strftime("%Y%m%d_%H%M%S")}.log', 'w') \
            if LOGS.exists() else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(argv, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT)
        except Exception as e:
            return refuse('insert_failed_to_start', str(e))
        fb = DriveMate.Feedback(); fb.stage = 'inserting'
        while self.proc.poll() is None:
            fb.z_above_seat = float('nan')
            gh.publish_feedback(fb)
            time.sleep(0.5)
        rc = self.proc.returncode; self.proc = None
        out = trace_outcome(obj, t0)
        rest = self.resting_tcp()
        if rest is not None and out.get('seat_xyz') is not None:
            out = judge_at_rest(out, rest)
        res.stage_reached = out['stage_reached']; res.seated = bool(out.get('seated', False))
        res.trace_path = out.get('trace_path', '')
        if 'slide' in out:
            res.slide = out['slide']; res.measured_correction = out['measured_correction']
        res.refusal = ''
        self.get_logger().info(f'{g.mate_id}: rc={rc} stage={res.stage_reached} seated={res.seated} '
                               f'fsm={out.get("fsm_outcome")} correction={out.get("measured_correction")}')
        if gh.is_cancel_requested:
            gh.canceled()
        else:
            gh.succeed()
        return res


def main():
    rclpy.init()
    n = DriveMateServer()
    ex = MultiThreadedExecutor(); ex.add_node(n)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
