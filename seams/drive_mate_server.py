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
                          with |tcp_z - predicted seat z| < insert_outcome.SEATED_DZ_M. The trace alone cannot say it (its seat
                          detector ends logging on arrival, still descending); its own reading is kept as trace_stage
          stage_reached   seat | wedged (still, 0 < dz <= 7 mm, within 5 mm of the aim: on a channel edge) | above |
                          below | unsettled (the trace ends still moving) | no_trace.  Checked 2026-10-20 against the
                          last 40 FMB1 inserts: 38 agree with the FSM label, 2 ended within 1 mm but moving
          slide           end TCP xy - predicted seat xy (what the search had to correct)
          measured_correction  [dx, dy, dz] of the end TCP vs the prediction
          trace_path      the insert's raw CSV
          refusal         mate_unknown | assembly_unbound | assembly_ambiguous | target_frame | held_frame |
                          held_unmatched | insert_failed_to_start | robot_stopped | corrections
                          | DRIVE_MATE_LATCHED: <why>   the server is latched (below); nothing was started
                          | INSERT_GATE: <why>          this goal's insert (or align) left the arm uncommandable:
                                                        dashboard stop sent, the server latched
                          | insert_rc: <rc> (<error>)   translate_object exited non-zero; the arm is commandable
                          | align_failed: <why>         the align step failed on a commandable arm; no insert
  cancel  SIGINT to the insert (its force-mode cleanup runs), SIGTERM after CHILD_STOP_GRACE_S (60 s) - never SIGKILL
          in force mode.  A cancelled goal LATCHES the server.  With the program already stopped (stop_run.sh's step 0)
          the server sends no stop - the canceller owns the restore (review r2 P2-A); with the program running or
          unknown (a Ctrl-C'd replay, a client timeout) it runs the full gate, stop included (review r3 P1).

The post-insert gate and the latch (lead 2026-11-20, w8-drivemate-gate).  After EVERY motion subprocess this server
starts (the align step and the insert), whatever the client: ablations.cell_io.post_insert_gate on translate_object's
own result (its __RESULT_JSON__, so a prismatic INSERT_RESTORE_FAILED is typed) and the robot's safety status.  The
gate sends the dashboard stop when the controllers are left switched (scaled inactive / passthrough or force_mode
active, or unreadable: fail closed) and the server LATCHES: every later goal is refused DRIVE_MATE_LATCHED, before
anything starts, until the server restarts or the latch is reset.  The reset is a DELIBERATE operator step (review r2
P2-C): std_srvs/Trigger /ur5e/drive_mate/reset_latch, after stop_run.sh and after the part is released and lifted; it
clears only when the safety is NORMAL / REDUCED, the controllers read restored and the external-control program runs.
/ur5e/drive_mate/latch_status (Trigger: success = not latched, message = the reason) is the read-only status that
stop_run.sh prints and that a SEAMS_DRIVE=1 replay and RealIO check at startup, refusing before any motion.
After EVERY motion child the server also latches when the program is not running (whatever the child reported: a
child that sent its own stop, review r2 P3-A).  A non-zero exit with the arm commandable is not a
latch: it is carried in `refusal` (insert_rc) and `seated` stays the trace's verdict, as before (the replay treats a
seated insert_rc as a success; the arm-commandable gate, the certifier and the per-seat check follow it).
Round 2 (review r1): the gate runs FIRST, the moment the child exits (before the resting-pose wait and the trace),
so a stop is never late; ANY exception in or after a motion child fails closed (gate crashed: the latch, and the
dashboard stop unless the controllers read restored); the align's own dashboard stop (drive_align: __ALIGN__
dashboard_stopped, non-zero exit) latches; a failed align (non-zero exit, no / malformed __ALIGN__, an error or a
safety stop in it) refuses align_failed and the insert does not start; an unreadable safety status fails closed;
the safety rule is cell_io's (NORMAL / REDUCED ok).  reset_latch answers at once when not latched and runs on its own
callback group.

Known debt, recorded here rather than hidden: the insert routes by part name (translate_object: lines -> prismatic,
U parts -> compliant spiral) and adds a per-part seat-z offset (PER_OBJECT_BASE_OFFSET_M) - the seam rule says
stages/routes come from CAD. Both live inside the primitive this server wraps; the next step is deriving the route
from the CAD stages (insert_stages.py) instead.
"""
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
from seams.insert_outcome import outcome_of  # noqa: E402
ACTION = '/ur5e/drive_mate'
B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))
LOGS = REPO / 'compliant_insertion_studio' / 'logs'
HELD_AXIS_TOL_M = 0.006          # a CAD grasp point this close to the tool axis is the grasp
REST_WINDOW_S, REST_STILL_M = 0.5, 0.0005    # the arm at rest after the insert: < 0.5 mm over 0.5 s


SAFETY_OK = ('NORMAL', 'REDUCED')      # cell_io._watch_safety's rule; an unreadable status (None) fails closed


LATCH_FILE = REPO / 'ablations' / 'logs' / 'drive_mate_latch.json'


class InsertLatch:
    """The server's latch: tripped by a failed gate, cleared ONLY by a checked, deliberate reset.  Persisted in
    LATCH_FILE (written atomically, removed only by a successful reset), so a server crash, an OOM kill, a re-login or a
    restart cannot clear it silently (review r3, lead ruling 2026-11-20): a new server starts latched from the file."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else LATCH_FILE
        self.reason = None
        try:
            self.reason = json.load(open(self.path)).get('reason') or None
        except FileNotFoundError:
            pass
        except Exception as e:                   # unreadable: fail closed
            self.reason = f'latch file {self.path} unreadable ({e!r})'

    def trip(self, reason):
        self.reason = self.reason or reason
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix('.tmp')
            with open(tmp, 'w') as fh:
                json.dump({'reason': self.reason, 'since': time.time()}, fh)
                fh.flush(); os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except Exception:
            pass                                  # the in-memory latch holds regardless

    def refusal(self):
        return f'DRIVE_MATE_LATCHED: {self.reason}' if self.reason else ''

    def reset(self, safety, restored, program_running):
        """Clear when the robot's safety is NORMAL / REDUCED, the controllers read restored and the external-control
        program runs (after stop_run.sh); else why not."""
        if self.reason is None:
            return ''
        if safety not in SAFETY_OK:
            return f'not reset: robot safety {safety or "unreadable"} (clear the stop, run stop_run.sh)'
        if not restored[0]:
            return f'not reset: controllers {restored[1]} (run stop_run.sh first)'
        if program_running is not True:
            return ('not reset: the external-control program is not running '
                    f'({"unknown" if program_running is None else "stopped"}; run stop_run.sh first)')
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            return f'not reset: the latch file {self.path} could not be removed ({e!r})'
        self.reason = None
        return ''


def result_of(text):
    """translate_object's own result, the last __RESULT_JSON__ block of its output; {} when none or not an object."""
    i = text.rfind('__RESULT_JSON__')
    if i < 0:
        return {}
    body = text[i + len('__RESULT_JSON__'):]
    j = body.find('__END_RESULT_JSON__')
    try:
        r = json.loads(body[:j] if j >= 0 else body)
    except Exception:
        return {}
    return r if isinstance(r, dict) else {}


def align_result(text):
    """drive_align's __ALIGN__ result (the last such line), or None when there is none or it does not parse."""
    line = next((l for l in reversed(text.splitlines()) if l.startswith('__ALIGN__ ')), None)
    if line is None:
        return None
    try:
        r = json.loads(line[len('__ALIGN__ '):])
    except Exception:
        return None
    return r if isinstance(r, dict) else None


def gate_after(rc, text, kind='insert', host=None):
    """After a motion child (kind 'insert': translate_object; 'align': drive_align): (refusal, latch_reason).
    latch_reason set when the arm was left uncommandable (cell_io.post_insert_gate sends the dashboard stop), when the
    child sent its own dashboard stop (the align's __ALIGN__ dashboard_stopped), or when the safety status is
    unreadable (fail closed); refusal '' when the arm is commandable and rc is 0."""
    from ablations import cell_io
    host = host or os.environ.get('ROBOT_IP', '192.168.1.111')
    if kind == 'align':
        al = align_result(text) or {}
        if al.get('dashboard_stopped'):
            why = f"the align sent the dashboard stop: {al['dashboard_stopped']}"
            return f'INSERT_GATE: {why}', why
        result, name = {}, 'drive_align'
    else:
        result, name = dict(result_of(text)), 'translate_object'
    if rc and not result.get('error'):
        result['error'] = f'{name} exited {rc}'
    safety = cell_io.robot_safety_status(host)
    if safety is None:
        stopped = cell_io.dashboard_stop(host)
        why = (f"robot safety status unreadable after the {kind}; dashboard stop "
               f"{'confirmed' if stopped else 'NOT confirmed - use the pendant'}")
        return f'INSERT_GATE: {why}', why
    cell_io._SAFETY_STOP = None if safety in SAFETY_OK else safety
    why = cell_io.post_insert_gate(result, host)
    if why:
        return f'INSERT_GATE: {why}', why
    if rc:
        return f"{kind}_rc: {rc} ({result.get('error', '')})", None
    return '', None


class StartFailed(Exception):
    """A motion child could not be started: nothing moved."""


def vision_pose(pose, source=''):
    """A VisionPose in the robot base frame from ((x, y, z), (qx, qy, qz, qw)); sigma unknown (NaN), as measured here."""
    from fmb_seam_msgs.msg import VisionPose
    v = VisionPose(); v.header.frame_id = 'base'; v.header.stamp = rclpy.clock.Clock().now().to_msg()
    (v.pose.position.x, v.pose.position.y, v.pose.position.z) = (float(c) for c in pose[0])
    (v.pose.orientation.x, v.pose.orientation.y, v.pose.orientation.z, v.pose.orientation.w) = (float(c) for c in pose[1])
    v.sigma = [float('nan')] * 4; v.source = source
    return v


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


class DriveMateServer(Node):
    def __init__(self):
        super().__init__('ur5e_drive_mate')
        self.tcp = None; self.proc = None; self.busy = threading.Lock(); self.latch = InsertLatch()
        self.program_running = None
        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger
        reads = ReentrantCallbackGroup()            # the TCP and program-state reads never wait on the reset service
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', lambda m: setattr(self, 'tcp', m.pose), 10,
                                 callback_group=reads)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Bool, '/io_and_status_controller/robot_program_running',
                                 lambda m: setattr(self, 'program_running', bool(m.data)), latched,
                                 callback_group=reads)
        ActionServer(self, DriveMate, ACTION, self.execute, goal_callback=self.on_goal,
                     cancel_callback=self.on_cancel, callback_group=ReentrantCallbackGroup())
        status = MutuallyExclusiveCallbackGroup()
        self.create_service(Trigger, ACTION + '/reset_latch', self.on_reset_latch, callback_group=status)
        self.create_service(Trigger, ACTION + '/latch_status', self.on_latch_status, callback_group=status)
        self.get_logger().info(f'drive-mate served at {ACTION}')

    def on_reset_latch(self, request, response):
        """Clear the latch after stop_run.sh: only when the safety is NORMAL / REDUCED, the controllers read restored
        and the external-control program runs.  Not latched: answered at once, nothing asked."""
        if self.latch.reason is None:
            response.success, response.message = True, 'not latched'
            return response
        from ablations import cell_io
        was = self.latch.reason
        why = self.latch.reset(cell_io.robot_safety_status(os.environ.get('ROBOT_IP', '192.168.1.111')),
                               cell_io.controllers_restored(), self.program_running)
        if not why:
            cell_io._DASHBOARD_STOPPED = None            # the server process's copy (review r1 P3-7)
        response.success = not why
        response.message = why or f'latch cleared (was: {was})'
        self.get_logger().warn(f'reset_latch: {response.message}')
        return response

    def on_latch_status(self, request, response):
        """Read-only: success = not latched; message = the latch reason (stop_run.sh prints it; startups check it)."""
        response.success = self.latch.reason is None
        response.message = self.latch.reason or 'not latched'
        return response

    def _after_motion(self, gh, rc, text, kind):
        """The checks after a motion child, in order: a cancel (latch, no stop: the canceller owns it), the gate, then
        the program state (latched when not running, whatever the child reported).  (refusal, latch_reason)."""
        if gh.is_cancel_requested and self.program_running is False:
            # the program is already down: stop_run.sh's step 0 stopped it; it owns the restore (review r2 P2-A)
            why = f'cancelled during the {kind} with the program already stopped (stop_run.sh)'
            return f'DRIVE_MATE_CANCELLED: {why}', why
        refusal, latch_why = gate_after(rc, text, kind)
        if gh.is_cancel_requested:
            # a cancel with the program running or unknown (a Ctrl-C'd replay, a client timeout): the full gate above,
            # stop included; latched either way (review r3 P1)
            why = latch_why or f'cancelled during the {kind} (program running: gate passed)'
            return f'DRIVE_MATE_CANCELLED: {why}', why
        if not latch_why and self.program_running is False:
            latch_why = f'the external-control program was stopped during the {kind}'
            refusal = f'INSERT_GATE: {latch_why}'
        return refusal, latch_why

    def _fail_closed(self, why):
        """Any exception in or after a motion child: latch, and send the dashboard stop unless the controllers read
        restored.  Returns the latch reason."""
        from ablations import cell_io
        host = os.environ.get('ROBOT_IP', '192.168.1.111')
        try:
            ok = bool(cell_io.controllers_restored()[0])
        except Exception:
            ok = False
        reason = f'gate crashed: {why}'
        if not ok:
            try:
                stopped = cell_io.dashboard_stop(host)
            except Exception:
                stopped = False
            reason += f"; dashboard stop {'confirmed' if stopped else 'NOT confirmed - use the pendant'}"
        self.latch.trip(reason)
        self.get_logger().error(f'{reason} - the server is latched until stop_run.sh + reset_latch')
        return reason

    def _run_motion(self, gh, argv, env, log_path, stage):
        """A motion child in its own process group, polled with feedback.  Returns (rc, its output).  Popen failing:
        StartFailed (nothing moved).  Any exception while it runs: SIGINT to the group, SIGTERM after the child grace
        (never SIGKILL), then re-raised for the caller to fail closed."""
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as log:
            try:
                self.proc = subprocess.Popen(argv, cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True)       # its own group: on_cancel signals all of it
            except Exception as e:
                raise StartFailed(str(e)) from e
            p = self.proc
            try:
                fb = DriveMate.Feedback(); fb.stage = stage
                while p.poll() is None:
                    fb.z_above_seat = float('nan')
                    gh.publish_feedback(fb)
                    time.sleep(0.5)
                rc = p.returncode
            except BaseException:
                try:
                    os.killpg(p.pid, signal.SIGINT)
                    self._term_later(p)
                except Exception:
                    pass
                raise
            finally:
                self.proc = None
        return rc, open(log_path).read()

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
            # the whole insert process group, as a Ctrl-C would: the force-mode wrapper / touch_insert get it directly
            # (10-22: SIGINT to translate_object alone ended in its subprocess.run SIGKILLing the child 0.25 s later)
            os.killpg(p.pid, signal.SIGINT)
            threading.Thread(target=self._term_later, args=(p,), daemon=True).start()
        return CancelResponse.ACCEPT

    @staticmethod
    def _term_later(p):
        from primitives.shared.config import CHILD_STOP_GRACE_S   # translate_object's child grace (review r1 P3-5)
        try:
            p.wait(timeout=CHILD_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)                  # never SIGKILL in force mode

    def _sigma_gate(self, g, obj, base):
        """(servo?, note) - decision 2026-10-22: drive-mate moves on a vision read of the held part over its seat only when
        the target's xy sigma exceeds the part's CAD clearance (seams.clearance). The target's sigma is the VisionPose's;
        NaN or 0 (not stated) -> the base fit's stated repeatability (bindings.yaml sigma.base_icp.xy). SIGMA_GATE=0: always."""
        if os.environ.get('SIGMA_GATE', '1') != '1':
            return True, 'off (SIGMA_GATE=0) - servo'
        try:
            sg = [float(v) for v in list(g.target.sigma)[:2]]      # a numpy array from the message (run 73: 'or []' threw)
            # NaN or 0 = not stated: the message's default sigma is zeros (run 74: u_brown's target arrived 0.00 mm, the gate
            # skipped the servo and the insert missed) - no read is exact, so a zero never means "inside the clearance"
            s_xy = max(sg) if sg and all(np.isfinite(sg)) and min(sg) > 0 else float('nan')
            src = 'the target read'
            if not np.isfinite(s_xy):
                B = yaml.safe_load(open(REPO / 'seams' / 'bindings.yaml'))
                s_xy, src = float(B['sigma']['base_icp']['xy']), 'the base fit (bindings)'
            from seams.clearance import xy_clearance_m
            from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far
            clr = xy_clearance_m(base, obj, list(get_seated_so_far(base, obj)))
        except Exception as e:
            return True, f'not judged ({e}) - servo'
        servo = s_xy > clr
        return servo, (f'xy sigma {1000 * s_xy:.2f} mm ({src}) vs CAD clearance {1000 * clr:.2f} mm -> '
                       + ('servo' if servo else 'no servo, the read is inside the clearance'))

    def _align(self, gh, g, obj, base, gid, measured, held_world, env):
        """seams.drive_align as a motion child: (rc, output, log path).  Parsed by the caller AFTER the gate."""
        argv = ['python3', '-u', '-m', 'seams.drive_align', '--obj', obj, '--base', base, '--grasp-id', str(gid),
                '--measured', *[f'{v:.6f}' for v in (*measured[0], *measured[1])],
                '--held-quat', *[f'{v:.6f}' for v in held_world]]
        log_path = LOGS / f'drive_align_{obj}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        rc, text = self._run_motion(gh, argv, env, log_path, 'aligning')
        return rc, text, log_path

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

        if self.latch.reason:                                # nothing starts on a latched server
            return refuse(self.latch.refusal())
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
        measured = ((tp.position.x, tp.position.y, tp.position.z),
                    (tp.orientation.x, tp.orientation.y, tp.orientation.z, tp.orientation.w))
        # read fresh every goal: a server started before a config edit kept the old offset and aimed yellow 6 mm
        # deep while the planner (a new process) used the new one (2026-10-21)
        import importlib
        import primitives.shared.config as _cfg
        importlib.reload(_cfg)
        # the seat height (UNI, 10-23): learned from this part's own certified inserts (seams.seat_learn); none yet -> the
        # CAD seat (dz 0). SEAT_DZ_SOURCE=table: the old hand table (PER_OBJECT_BASE_OFFSET_M + settle)
        from seams import seat_learn as _sl
        if os.environ.get('SEAT_DZ_SOURCE', 'learned') == 'table':
            dz, dz_src = float(_cfg.get_object_base_offset_m(obj)[2]), 'table'
        else:
            dz_l, n_l = _sl.learned_dz(obj)
            dz, dz_src = (dz_l, f'learned from {n_l} certified inserts') if dz_l is not None else (0.0, 'CAD (nothing learned yet)')
        self.get_logger().info(f'{g.mate_id}: seat dz {1000 * dz:+.2f} mm ({dz_src})')
        env = os.environ.copy()
        env['GRASP_TOOL_OFFSET_M'] = f'{depth:.6f}'; env['ROS_MCP_MODE'] = 'real'
        env['PYTHONPATH'] = f"{REPO}:{env.get('PYTHONPATH', '')}"
        # align (AR3, decision 10-22: vision measures, drive-mate moves): on a free base bring the part over its seat
        # first - today's servo, run here instead of by the planner. Then the planner's corrections (follow / bias)
        # on the base the alignment arrived at, in the order the planner applied them before.
        aim, align = measured, {}
        servo, gate_note = self._sigma_gate(g, obj, base)
        self.get_logger().info(f'{g.mate_id}: sigma gate: {gate_note}')
        if not bool(getattr(_cfg, 'BASE_FIXED', True)) and os.environ.get('DRIVE_ALIGN', '1') == '1' and not servo:
            align = {'route': 'none (sigma gate: ' + gate_note + ')'}
        elif not bool(getattr(_cfg, 'BASE_FIXED', True)) and os.environ.get('DRIVE_ALIGN', '1') == '1':
            if gh.is_cancel_requested:
                gh.canceled(); return res
            try:
                rc_a, text_a, log_a = self._align(gh, g, obj, base, gid, measured, held_world, env)
                refusal, latch_why = self._after_motion(gh, rc_a, text_a, 'align')   # FIRST: the align moves the arm
            except StartFailed as e:
                return refuse('insert_failed_to_start', f'align: {e}')
            except Exception as e:
                return refuse('INSERT_GATE: ' + self._fail_closed(f'align: {e!r}'))
            if latch_why:
                self.latch.trip(latch_why)
                if refusal.startswith('DRIVE_MATE_CANCELLED'):
                    res.refusal = refusal
                    gh.canceled(); return res
                return refuse(refusal)
            align = align_result(text_a)
            if refusal or align is None or align.get('error') or align.get('safety'):
                why = (refusal or ('no __ALIGN__ result' if align is None else '')
                       or align.get('error') or f"robot {align.get('safety')}")
                return refuse(f'align_failed: {why}', log_a.name)            # the insert does not start
            align['log'] = str(log_a)
            self.get_logger().info(f"{g.mate_id}: aligned ({align.get('route')}, measured={align.get('measured')}) "
                                   f"- {log_a.name}")
            if align.get('aim_base'):
                aim = (tuple(align['aim_base'][0]), tuple(align['aim_base'][1]))
            if align.get('measured_base'):
                measured = (tuple(align['measured_base'][0]), tuple(align['measured_base'][1]))
            if align.get('grasp_depth_fix_m'):
                # the align step's touchdown measured the grasp depth: insert with it (seat, rim, entering heights)
                depth += float(align['grasp_depth_fix_m'])
                env['GRASP_TOOL_OFFSET_M'] = f'{depth:.6f}'
                self.get_logger().info(f"{g.mate_id}: grasp depth measured by the align touchdown: {1000 * depth:.1f} mm "
                                       f"({1000 * align['grasp_depth_fix_m']:+.1f})")
            tq = self.tcp.orientation                        # the wrist may have turned: the part turned with it
            held_world = (R.from_quat([tq.x, tq.y, tq.z, tq.w]) * R.from_quat(q_fo)).as_quat()
        try:
            corr = json.loads(g.corrections_json) if g.corrections_json else []
        except Exception as e:
            return refuse('corrections', str(e))
        if corr:
            from seams.aim import Cad, apply_corrections
            aim = apply_corrections(aim, corr, seat_in_base=Cad().seat_pos(base, obj))
        res.base_measured = vision_pose(measured, source=f"drive_align: {align.get('route', 'none')}")
        res.align_json = json.dumps(dict(align, corrections=[c.get('name') for c in corr],
                                         aim=[list(aim[0]), list(aim[1])]))
        argv = ['python3', '-u', '-m', 'primitives.translate_object', '--mode', 'real', '--object-name', obj,
                '--base-name', base, '--insert', '--insertion-type', 'compliant', '--grasp-id', str(gid),
                '--current-object-orientation', *[f'{v:.6f}' for v in held_world],
                '--final-base-pos', f'{aim[0][0]:.6f}', f'{aim[0][1]:.6f}', f'{aim[0][2] + dz:.6f}',
                '--final-base-orientation', *[f'{v:.6f}' for v in aim[1]]]
        # the part's yaw in the jaws, world z, folded to +-45 (a square grasp gives 0): the hover subtracts it
        tcp_yaw = float(R.from_quat([tq.x, tq.y, tq.z, tq.w]).as_euler('xyz')[2])
        part_yaw = float(R.from_quat(held_world).as_euler('xyz')[2])
        held_yaw_off = (np.degrees(part_yaw - tcp_yaw) + 45.0) % 90.0 - 45.0
        env['HELD_YAW_OFFSET_DEG'] = f'{held_yaw_off:.3f}'
        if align.get('route'):
            # the align step turned the wrist to the base yaw already: the insert's hover must not turn it again (run 28c:
            # line_green turned a further -9.09 deg by an Euler 'held yaw' of a part held at pitch 90 - noise - and jammed)
            env['HOVER_YAW_FIX'] = '0'; env['HELD_YAW_OFFSET_DEG'] = '0'
            held_yaw_off = 0.0
        # one attempt per goal, a wedged bar held where it stopped: retries and recoveries are the caller's (the
        # replay sets the same two at import)
        env['PRISMATIC_MAX_RETRIES'] = '1'; env['PRISMATIC_HOLD_IF_WEDGED'] = '1'
        self.get_logger().info(f'{g.mate_id}: {obj} into {base}, grasp {gid} at depth {1000 * depth:.1f} mm '
                               f'({1000 * off:.1f} mm off axis), in-hand yaw {held_yaw_off:+.2f} deg')
        if gh.is_cancel_requested:                       # a cancel between the align and the insert (r1 P3-5)
            gh.canceled(); return res
        t0 = time.time()
        log_path = LOGS / f'drive_mate_{obj}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        try:
            rc, text = self._run_motion(gh, argv, env, log_path, 'inserting')
            refusal, latch_why = self._after_motion(gh, rc, text, 'insert')   # FIRST, the moment the child exits
        except StartFailed as e:
            return refuse('insert_failed_to_start', str(e))
        except Exception as e:
            res.refusal = 'INSERT_GATE: ' + self._fail_closed(f'insert: {e!r}')
            gh.abort(); return res
        if latch_why:
            self.latch.trip(latch_why)
            self.get_logger().error(f'{g.mate_id}: {refusal} - the server is latched until stop_run.sh + reset_latch')
        res.refusal = refusal
        try:
            # judged once, on the trace the insert named (seams.insert_outcome); judged where the arm came to rest
            o = outcome_of(text, obj, t0, rest=self.resting_tcp())
            if o.found_by != 'result':
                self.get_logger().warn(f'{g.mate_id}: the insert named no trace - {o.found_by}'
                                       + (f' (newest since the goal: {o.meta_path})' if o.meta_path else ''))
            out = o.as_dict()
            res.stage_reached = out['stage_reached']; res.seated = bool(out.get('seated', False))
            res.trace_path = out.get('trace_path', '')
            if out.get('slide') is not None:
                res.slide = out['slide']; res.measured_correction = out['measured_correction']
                try:
                    _sl.record_insert(obj, dz, float(out['measured_correction'][2]), res.seated)   # ok set by the certifier
                except Exception as e_:
                    self.get_logger().warn(f'{g.mate_id}: seat record not written ({e_})')
        except Exception as e:
            if not latch_why:                    # r2 P3-B: the gate passed; say so, so nobody hunts for a crash
                res.refusal = 'INSERT_GATE: ' + self._fail_closed(
                    f'trace unreadable ({e!r}); arm checked commandable by the gate')
            if res.refusal.startswith('DRIVE_MATE_CANCELLED'):
                gh.canceled()
            else:
                gh.abort()
            return res
        self.get_logger().info(f'{g.mate_id}: rc={rc} stage={res.stage_reached} seated={res.seated} '
                               f'fsm={out.get("fsm_outcome")} correction={out.get("measured_correction")}'
                               + (f' refusal={refusal}' if refusal else ''))
        if latch_why:
            if refusal.startswith('DRIVE_MATE_CANCELLED'):
                gh.canceled()
            else:
                gh.abort()
            return res
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
