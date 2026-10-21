#!/usr/bin/env python3
"""Replay an Assembly_*.json tool sequence against the LIVE REAL ROBOT.

This is the real-mode counterpart to ablations/replay_verify.py (which is
sim-only and runs inside the unified_api/orchestrator environment). This
script does NOT depend on unified_api — it spawns each primitive as a
subprocess and parses its __RESULT_JSON__.

ASSUMES the robot stack is already up:
  - launch_robot.sh real           (UR driver + controllers, pendant in Play)
  - gripper_control bridge running (socat → /tmp/ttyUR → OnRobot RG2)
  - launch_camera.sh --background  (aruco_camera_localizer publishing
                                    /objects_poses_real)

This script does NOT launch any nodes. If anything's missing, the first
primitive will fail and the script will surface the error.

Usage:
  # Run the full assembly:
  python3 ablations/replay_real_assembly.py \
    --assembly-json ablations/ground_truth_resources/Assembly_fmb_assembly_1_results.json

  # Run only one object's tool_sequence (iterative debugging):
  python3 ablations/replay_real_assembly.py \
    --assembly-json <path> --only u_brown

  # Per-object xy offset (m) injected into the insert step's base pose:
  python3 ablations/replay_real_assembly.py \
    --assembly-json <path> --only u_brown --base-offset-xy 0.0015 -0.002

  # Skip the place_down + re-grasp middle of an object's sequence (operator
  # is feeding the part directly to insert):
  python3 ablations/replay_real_assembly.py \
    --assembly-json <path> --only u_brown --skip-to insert

  # Dry run — print the planned subprocess argv without executing:
  python3 ablations/replay_real_assembly.py \
    --assembly-json <path> --only u_brown --dry-run

The script chains current-object-orientation between rotate_object and
translate_object steps automatically (mirrors run_assembly_step.py's
chaining logic). All sim-mode tool calls in the JSON are rewritten to
real-mode at execution time.

Reference: derived from ablations/replay_verify.py (sim) + the tool-call
parser there. Real-mode wiring uses the compliant_insert path that
translate_object.py auto-dispatches to under --insertion-type compliant
(the default for real-mode insert as of 2026-05-07).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
from scipy.spatial.transform import Rotation as R_  # noqa: E402
LOG_ROOT = REPO_ROOT / "ablations" / "logs"

# Make `primitives.*` importable from this script's own Python process.
# Subprocesses get this via make_env() but the build_argv path needs the
# config constants in-process to compute base-offset overrides.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Set when a --stop-before OR --stop-after token triggers, so main() can skip
# the cleanup move_home (the whole point of "pause" is to leave the robot at
# the halted pose for operator inspection or follow-on motion).
_STOP_BEFORE_FIRED = False
_STOP_AFTER_FIRED = False

# Module-global handle to the currently running subprocess so the SIGINT
# handler can kill it (its whole process group) when the operator hits Ctrl-C.
_ACTIVE_PROC: Optional[subprocess.Popen] = None
_INTERRUPTED = False
_SAFETY_STOP: Optional[str] = None     # set when the robot leaves NORMAL safety: the run halts, no cleanup
_HOLDING_SEATED = False                # the gripper is closed on a part that is seated: never lift


def robot_safety_status(host: str = '192.168.1.111', timeout: float = 1.5) -> Optional[str]:
    """'NORMAL', 'PROTECTIVE_STOP', ... from the UR dashboard server (port 29999), None if unreachable.
    Independent of ROS, so it works while the driver or a primitive is wedged."""
    import socket
    try:
        with socket.create_connection((host, 29999), timeout=timeout) as s_:
            s_.settimeout(timeout)
            s_.recv(1024)                                      # banner
            s_.sendall(b'safetystatus\n')
            return s_.recv(1024).decode(errors='replace').strip().split(':')[-1].strip()
    except Exception:
        return None


def _sigint_handler(signum, frame):
    """On Ctrl-C, kill the active subprocess group and abort.

    Without this the replay script's own KeyboardInterrupt won't propagate to
    the child primitive's process tree, leaving the robot in motion while
    the orchestrator crashes — a recipe for protective-stops.
    """
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n\n!!! SIGINT received — killing active subprocess and aborting replay !!!", flush=True)
    if _ACTIVE_PROC is not None:
        # Wait for the WHOLE process group, not just the direct child: an insert runs
        # translate_object -> compliant_insert, and the wrapper's SIGINT handler stops
        # force mode and switches controllers back, which outlived the old 3 s grace.
        # Exiting early left it finishing alone and, twice on 2026-10-07, left the
        # robot's external-control program stopped.
        try:
            pgid = os.getpgid(_ACTIVE_PROC.pid)
        except Exception:
            pgid = None

        def _group_alive():
            if pgid is None:
                return _ACTIVE_PROC.poll() is None
            try:
                os.killpg(pgid, 0)
                return True
            except ProcessLookupError:
                return False
            except Exception:
                return _ACTIVE_PROC.poll() is None

        def _wait_group(seconds):
            end = time.time() + seconds
            while time.time() < end and _group_alive():
                try:
                    _ACTIVE_PROC.wait(timeout=0.2)   # reap the direct child
                except Exception:
                    time.sleep(0.2)
            return not _group_alive()

        for sig, grace in ((signal.SIGINT, 20.0), (signal.SIGTERM, 10.0)):
            if not _group_alive():
                break
            try:
                if pgid is not None:
                    os.killpg(pgid, sig)
                else:
                    _ACTIVE_PROC.send_signal(sig)
            except Exception:
                pass
            if _wait_group(grace):
                break
        # Never SIGKILL: a SIGKILL mid force-mode leaves nothing to switch it off.
        if _group_alive():
            print("!!! primitive process group still alive after SIGINT+SIGTERM — "
                  "run compliant_insertion_studio/scripts/stop_run.sh", flush=True)
    sys.exit(130)  # 128 + SIGINT


# ---------------------------------------------------------------------------
# Tool-call parsing — matches replay_verify.py format
# ---------------------------------------------------------------------------

_RESULT_RE = re.compile(r'__RESULT_JSON__\s*(.*?)\s*__END_RESULT_JSON__', re.DOTALL)


def parse_tool_call(call_str: str) -> Tuple[str, dict]:
    """Parse 'ros-mcp-server__tool(key='val', ...)' into (tool_name, kwargs)."""
    call_str = re.sub(r'\s*\[.*?\]\s*$', '', call_str)  # strip trailing [FAILED] etc.
    paren = call_str.index('(')
    raw_name = call_str[:paren]
    args_str = call_str[paren + 1:call_str.rindex(')')]
    if raw_name.startswith('ros-mcp-server__'):
        tool_name = raw_name[len('ros-mcp-server__'):]
    elif raw_name.startswith('ros_mcp_server__'):
        tool_name = raw_name[len('ros_mcp_server__'):]
    else:
        tool_name = raw_name
    kwargs = {}
    if args_str.strip():
        try:
            kwargs = eval(f"dict({args_str})")
        except Exception as e:
            raise RuntimeError(f"Failed to parse args of {raw_name!r}: {e}") from e
    return tool_name, kwargs


def extract_result_json(text: str) -> Optional[dict]:
    m = _RESULT_RE.search(text)
    if not m:
        return None
    payload = m.group(1).strip()
    # Prefer the LAST __RESULT_JSON__ block in the output if multiple were emitted
    last_match = None
    for found in _RESULT_RE.finditer(text):
        last_match = found
    if last_match:
        payload = last_match.group(1).strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

_SNAPSHOT_HELPER = r"""
import json, sys, time, rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

class _Snap(Node):
    def __init__(self):
        super().__init__('replay_snapshot')
        self.joints = None
        self.tcp = None
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', self._tcp, 10)
    def _js(self, m):
        self.joints = {'name': list(m.name),
                       'position': [float(x) for x in m.position],
                       'stamp_s': m.header.stamp.sec + m.header.stamp.nanosec*1e-9}
    def _tcp(self, m):
        p, q = m.pose.position, m.pose.orientation
        self.tcp = {'xyz': [p.x, p.y, p.z],
                    'quat_xyzw': [q.x, q.y, q.z, q.w],
                    'stamp_s': m.header.stamp.sec + m.header.stamp.nanosec*1e-9}

rclpy.init()
n = _Snap()
deadline = time.time() + 1.5
while time.time() < deadline and (n.joints is None or n.tcp is None):
    rclpy.spin_once(n, timeout_sec=0.05)
print(json.dumps({'joints': n.joints, 'tcp': n.tcp}))
n.destroy_node()
rclpy.shutdown()
"""


def snapshot_robot_state(timeout_s: float = 2.0) -> dict:
    """Grab a single sample of /joint_states + /tcp_pose_broadcaster/pose by
    spawning a tiny rclpy subscriber subprocess. Returns {joints: ..., tcp: ...}
    or {error: ...} on failure. Out-of-process so we don't conflict with any
    long-running rclpy contexts in the orchestrator."""
    try:
        proc = subprocess.run(
            ['python3', '-c', _SNAPSHOT_HELPER],
            capture_output=True, text=True, timeout=timeout_s, env=make_env(),
        )
        if proc.returncode != 0:
            return {'error': f'snapshot rc={proc.returncode}',
                    'stderr_tail': proc.stderr.splitlines()[-3:] if proc.stderr else []}
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else '{}'
        return json.loads(line)
    except subprocess.TimeoutExpired:
        return {'error': f'snapshot timeout after {timeout_s}s'}
    except Exception as e:
        return {'error': f'snapshot failed: {e}'}


# ---------------------------------------------------------------------------
# Staging landing check (2026-10-07)
# ---------------------------------------------------------------------------
# The recorded with_regrasp sequences place the part down once and assume it lands on
# the face that lets the second pass insert. 2026-10-07: u_brown was laid on its side
# and toppled back upright; the regrasp took it anyway and the insert ran with the
# wrong face held. A landing is just another start state (planning/sequence_planner.py):
# after the staging regrasp observes it from above, ask the planner's question — can a
# single pass (pick -> rotate -> insert) seat it from here? If not, run another
# place-down pass.

STAGING_MAX_RETRIES = 2   # extra place-down passes before giving up
GRIP_WIDTH_TOL_MM = 8.0   # a close ending further than this from the grasp's own width took the wrong thing


def staging_single_pass_ok(obj: str, base_name: str, observed_quat):
    """(feasible, up_face_err_deg) for a part observed at the staging area, or None if
    it cannot be evaluated (no observation / no assembly entry)."""
    from compliant_insertion_studio.planning.sequence_planner import (
        quat_to_matrix, single_pass_feasible)
    from primitives.shared.config import DEFAULT_BASE_ORIENTATION
    from primitives.shared.fold_symmetry import load_symmetry_data
    from utils.data_path_finder import (find_assembly_json_by_base_name,
                                        get_assembly_data_dir, get_symmetry_dir)
    if observed_quat is None or not base_name:
        return None
    try:
        jf = find_assembly_json_by_base_name(base_name, str(get_assembly_data_dir()))
        comps = json.load(open(jf))['components']
        tq = next(c for c in comps if c['name'] == obj)['rotation']['quaternion']
        R_target = (quat_to_matrix(DEFAULT_BASE_ORIENTATION)
                    @ quat_to_matrix([tq[k] for k in 'xyzw']))
        fold = load_symmetry_data(obj, str(get_symmetry_dir()))
        ok, err, _ = single_pass_feasible(quat_to_matrix(observed_quat), R_target, fold)
        return ok, err
    except Exception:
        return None


# ---------------------------------------------------------------------------
# --derive-sequence: plan 1 or 2 passes from the observed start pose
# (ported 2026-10-07 from the 2026-08-17 session; planning/sequence_planner.py)
# ---------------------------------------------------------------------------
_OBSERVE_ORIENTATION_HELPER = r"""
import json, sys, time, rclpy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped

want = sys.argv[1]
deadline_s = float(sys.argv[2])
want_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 3
STILL_WINDOW_S = 0.8     # a pose counts only if the TCP did not move (> 0.5 mm) in this window before it was
                         # published: the localizer stamps at publish, frames arrive 150-650 ms late (depth_timing
                         # 2026-10-20) and poses from frames taken while moving are ~10 mm off

rclpy.init()
n = rclpy.create_node('replay_observe_orientation')
samples = []; pending = []; tcp = []

def cb(msg):
    for t in msg.transforms:
        if t.child_frame_id == want:
            q = t.transform.rotation
            p = t.transform.translation
            pending.append((time.time(), {'quat_xyzw': [q.x, q.y, q.z, q.w], 'xyz': [p.x, p.y, p.z]}))

def on_tcp(m):
    tcp.append((time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z))
    del tcp[:-3000]

def settle():
    keep = []
    for t_pub, smp in pending:
        if not tcp or tcp[-1][0] < t_pub:                 # TCP history must cover the publish time
            keep.append((t_pub, smp)); continue
        win = [c for c in tcp if t_pub - STILL_WINDOW_S <= c[0] <= t_pub]
        if len(win) >= 10 and max(((c[1]-win[-1][1])**2 + (c[2]-win[-1][2])**2 + (c[3]-win[-1][3])**2) ** 0.5 for c in win) <= 0.0005:
            samples.append(smp)
    pending[:] = keep

n.create_subscription(TFMessage, '/objects_poses_real', cb, 10)
n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', on_tcp, 100)
deadline = time.time() + deadline_s
while time.time() < deadline and len(samples) < want_samples:
    rclpy.spin_once(n, timeout_sec=0.02)
    settle()
print(json.dumps({'object': want, 'samples': samples,
                  'waited_s': round(deadline_s - (deadline - time.time()), 2)}))
n.destroy_node()
rclpy.shutdown()
"""


_OBSERVE_MANY_HELPER = r"""
import json, sys, time, rclpy
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseStamped
want = set(sys.argv[1].split(',')); deadline_s = float(sys.argv[2]); n_want = int(sys.argv[3])
SETTLE_S = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
STILL_WINDOW_S = 0.8
rclpy.init(); n = rclpy.create_node('replay_observe_many')
samples = {}; pending = []; tcp = []; last_new = [time.time()]
def cb(msg):
    for t in msg.transforms:
        if t.child_frame_id in want:
            q = t.transform.rotation; p = t.transform.translation
            pending.append((time.time(), t.child_frame_id, {'quat_xyzw': [q.x, q.y, q.z, q.w], 'xyz': [p.x, p.y, p.z]}))
def on_tcp(m):
    tcp.append((time.time(), m.pose.position.x, m.pose.position.y, m.pose.position.z)); del tcp[:-3000]
def settle():
    keep = []
    for t_pub, name, smp in pending:
        if not tcp or tcp[-1][0] < t_pub:
            keep.append((t_pub, name, smp)); continue
        win = [c for c in tcp if t_pub - STILL_WINDOW_S <= c[0] <= t_pub]
        if len(win) >= 10 and max(((c[1]-win[-1][1])**2 + (c[2]-win[-1][2])**2 + (c[3]-win[-1][3])**2) ** 0.5 for c in win) <= 0.0005:
            if name not in samples:
                last_new[0] = time.time()
            samples.setdefault(name, []).append(smp)
    pending[:] = keep
n.create_subscription(TFMessage, '/objects_poses_real', cb, 10)
n.create_subscription(PoseStamped, '/tcp_pose_broadcaster/pose', on_tcp, 100)
t0 = time.time(); deadline = t0 + deadline_s
while time.time() < deadline:
    rclpy.spin_once(n, timeout_sec=0.02); settle()
    full = [k for k in samples if len(samples[k]) >= n_want]
    # done when every name seen so far has its samples and no new name has appeared for SETTLE_S
    if samples and len(full) == len(samples) and time.time() - last_new[0] >= SETTLE_S:
        break
print(json.dumps({'samples': samples, 'waited_s': round(time.time() - t0, 2)}))
n.destroy_node(); rclpy.shutdown()
"""


def observe_many(names, timeout_s=8.0, n_samples=12, settle_s=2.0) -> dict:
    """{name: [samples]} for every name the camera sees now, from ONE still-camera window (Speed T1, 2026-10-21:
    names observed one by one each waited out their own timeout when not in view - 8 s per absent part)."""
    try:
        proc = subprocess.run(['python3', '-c', _OBSERVE_MANY_HELPER, ','.join(names), str(timeout_s), str(n_samples),
                               str(settle_s)], capture_output=True, text=True, timeout=timeout_s + 15, env=make_env())
        return json.loads(proc.stdout.strip().splitlines()[-1]).get('samples', {})
    except Exception:
        return {}


def observe_object_orientation(object_name: str, timeout_s: float = 30.0, n_samples: int = 3) -> dict:
    """Observe a part's resting pose on /objects_poses_real. Waits rather than sampling
    once: the localizer goes quiet while the arm moves, so absence of a detection means
    "not seen yet", not "not there". The camera only sees what is under the arm — put the
    arm over the part first (home view or move_to_clear_area --fix-orientation)."""
    try:
        proc = subprocess.run(
            ['python3', '-c', _OBSERVE_ORIENTATION_HELPER, object_name, str(timeout_s), str(n_samples)],
            capture_output=True, text=True, timeout=timeout_s + 10, env=make_env(),
        )
        if proc.returncode != 0:
            return {'error': f'observe rc={proc.returncode}'}
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else '{}'
        return json.loads(line)
    except Exception as e:
        return {'error': f'observe failed: {e}'}


LAST_BASE_POSE = REPO_ROOT / 'ablations/logs/last_base_pose.json'


def _save_base_pose(base_name, ob):
    if base_name == RUN_BASE.get('name'):
        RUN_BASE['pose'] = ob                     # the applied read: picks reuse it (Speed T1)
    try:
        d = json.load(open(LAST_BASE_POSE)) if LAST_BASE_POSE.exists() else {}
        d[base_name] = [list(map(float, ob[0])), list(map(float, ob[1])), time.time()]
        json.dump(d, open(LAST_BASE_POSE, 'w'), indent=1)
    except Exception:
        pass


def _last_base_pose(base_name):
    try:
        p, q, _ = json.load(open(LAST_BASE_POSE))[base_name]
        return tuple(p), tuple(q)
    except Exception:
        return None


def camera_brightness(timeout=3.0):
    """Mean of one colour frame (0-255), None if no frame."""
    try:
        proc = subprocess.run(['python3', '-c', _BRIGHTNESS_HELPER], capture_output=True, text=True, timeout=timeout + 10,
                              env=make_env())
        return float(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


_BRIGHTNESS_HELPER = r"""
import time, rclpy, numpy as np
from sensor_msgs.msg import Image
rclpy.init(); n = rclpy.create_node('brightness'); g = {}
n.create_subscription(Image, '/camera/camera/color/image_raw', lambda m: g.__setitem__('c', m), 1)
t = time.time()
while 'c' not in g and time.time() - t < 3: rclpy.spin_once(n, timeout_sec=0.1)
print(float(np.frombuffer(g['c'].data, np.uint8).mean()) if 'c' in g else -1)
"""
LOW_LIGHT_BRIGHTNESS = 40.0      # below: markers need close views (0.40 m saw line_red 5/5 at brightness ~5)


def observe_base_pose(base_name: str, timeout_s: float = 15.0):
    """The base's pose as the camera sees it, for the inserts (2026-10-14).

    Grasps aim at what the camera sees; inserts used to aim at a hand-tuned
    DEFAULT_BASE_POSITION (~5 mm off the measured base) plus per-part xy fudges. Aiming
    the inserts at the camera-observed base makes the chain self-consistent: a bias the
    camera/arm puts on a part's grasp also sits on its insert target, so it cancels.
    Returns ((x, y, z), (qx, qy, qz, qw)) or None."""
    obs = observe_object_orientation(base_name, timeout_s, n_samples=15)
    S = obs.get('samples') or []
    if obs.get('error') or len(S) < 5:
        # markers not readable (low light, 2026-10-20: brightness 4.6/255 and no marker anywhere, while the depth
        # CAD fit still read the base at 1.27 mm rms): fall back to the depth fit seeded from the last base pose
        if os.environ.get('BASE_FROM') == 'icp' and str(base_name).startswith('base'):
            seed = _last_base_pose(base_name)
            if seed is not None:
                try:
                    from base_icp import read_icp
                    oi, info = read_icp(base_name, seed)
                    print(f"  [base] {base_name}: markers not read -> depth CAD fit from the last pose: {json.dumps(info)}", flush=True)
                    return oi
                except Exception:
                    return None
        return None
    import numpy as _np
    P = _np.median(_np.array([s_['xyz'] for s_ in S]), axis=0)
    Q = _np.array([s_['quat_xyzw'] for s_ in S]); Q *= _np.sign(Q @ Q[0])[:, None]
    q = Q.mean(axis=0); q /= _np.linalg.norm(q)
    return tuple(float(v) for v in P), tuple(float(v) for v in q)



ASSEMBLY_STATE = REPO_ROOT / 'ablations/logs/assembly_state.json'   # {part: {verified, t, how}}
# Parts a later part hides from the camera once seated (operator, 2026-10-17: after yellow, line_green
# is not visible — accept it from the ledger if it was verified before yellow covered it).
OCCLUDED_BY = {'line_green': ('inverted_u_yellow',)}
VERIFY_READS = 3                 # verify_assembly readings per part; majority (of detections) decides
VERIFY_MAX_READS = 8             # "not detected" reads extend the window up to this many


def _ledger():
    try:
        return json.load(open(ASSEMBLY_STATE))
    except Exception:
        return {}


def forget_verified(obj: str) -> None:
    d = _ledger()
    if d.pop(obj, None) is not None:
        json.dump(d, open(ASSEMBLY_STATE, 'w'), indent=1)


VERIFY_TILT_MAX_DEG = 5.0        # a seated part cannot be tilted: a reading this tilted is a bad pose estimate


def _not_detected(r) -> bool:
    """No usable reading: not detected, OR a pose estimate tilted > VERIFY_TILT_MAX_DEG — seated u_brown
    seen edge-on from the base view reads roll -9.5 / pitch 10.4 deg and 8-10 mm xy, while home reads it at
    ~2 mm (2026-10-19: a good seat, Delta +0.21 mm, failed the check at 10.2 mm). Such readings do not
    vote; with none usable, verify_seated takes the second look from home."""
    if r.get('detected') is False or 'not detected' in str(r.get('error', '')):
        return True
    o = r.get('orientation_error_deg') or {}
    return max(abs(float(o.get('roll') or 0)), abs(float(o.get('pitch') or 0))) > VERIFY_TILT_MAX_DEG


RUN_BASE = {'name': 'base1'}      # the base this run assembles on (set in main from the recorded sequence)


def _read_part(o):
    """Up to VERIFY_MAX_READS verify_assembly readings of `o`; returns the deciding one.
    A part fails only when most of its detections fail — one misread of a half-hidden marker
    stopped a run on a seated u_orange (2026-10-17), while a part that is really off (u_brown on
    the table, 279 mm) fails every reading. Only readings that DETECTED the part vote: "not
    detected" is no evidence either way — the localizer drops whole frames for a second or two
    (2026-10-17: a seated u_brown, clearly in view, missed by two reads 1.5 s apart, and the run
    stopped). Those just extend the window; never detected in the window -> a not-detected result."""
    reads = []
    for k_ in range(VERIFY_MAX_READS):
        r = run_subprocess(['python3', '-u', '-m', 'queries.verify_assembly', '--object-name', o,
                            '--base-name', RUN_BASE['name'], '--mode', 'real', '--xy-only'],
                           label=f'verify_assembly {o} ({k_ + 1})', timeout_s=40)
        reads.append(r)
        det = [x for x in reads if not _not_detected(x)]
        if len(det) == 1 and det[0] is r and r.get('result') == 'success':
            break                                       # a clean first detection is enough
        if sum(x.get('result') == 'success' for x in det) * 2 > VERIFY_READS:
            break
        if sum(x.get('result') != 'success' for x in det) * 2 > VERIFY_READS:
            break
        if _not_detected(r):
            time.sleep(1.0)
    det = [x for x in reads if not _not_detected(x)]
    good = [x for x in det if x.get('result') == 'success']
    if not det:
        return reads[-1]
    return good[0] if len(good) * 2 > len(det) or (det[0].get('result') == 'success' and len(det) == 1) \
        else next(x for x in det if x.get('result') != 'success')


def _staging_view(log_fh=None):
    """Camera over the staging region (staging_layout.STAGING_VIEW, top-down, empty gripper at safe
    height or above): where the generated-layout disassembly puts parts — home sees x <= ~0.42 only."""
    from staging_layout import STAGING_VIEW, TOP_DOWN_Q
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                        *[f"{v:.4f}" for v in STAGING_VIEW], '--quat', *map(str, TOP_DOWN_Q)],
                       label='staging view', timeout_s=90, log_fh=log_fh)
    if r.get('result') != 'failure':
        time.sleep(1.5)                   # let the localizer catch up after the move
        return True
    return False


LAST_SEEN_XY = {}                 # part -> (x, y) from its latest observation (derive step, view checks)
PART_VIEW_Z = 0.52                # flange height for a look straight above one part (RGB only; depth not needed)
PART_VIEW_Z_LOW_LIGHT = 0.40      # in low light the markers need the closer view (2026-10-20)


def _seen_here(obj, timeout_s=5.0) -> bool:
    obs = observe_object_orientation(obj, timeout_s, n_samples=2)
    ok = bool(obs.get('samples')) and not obs.get('error')
    if ok:
        LAST_SEEN_XY[obj] = tuple(obs['samples'][-1]['xyz'][:2])
    return ok


PICK_FINGER_CLEAR_MM = 15.0       # the recorded grasp's open fingers must have this much room from base1 and the
                                  # other parts (base read scatters ~8 mm): yellow flat with its bridge grasp 5 mm from
                                  # base1's edge put a finger on the base -> PROTECTIVE_STOP (2026-10-19)


PICK_OPENING_MM = {}              # part -> pre-grasp opening chosen for the stance it lies in (see _pick_room)
PICK_OPEN_MAX_MM = 100.0
PICK_POSE_MARGIN_MM = 10.0       # open finger room per side at a pick: tonight's table reads were 5-10 mm off
PICK_CMD_TO_GAP_MM = 4.0         # commanded opening -> tip gap (39.8 commanded read 35.8, 2026-10-21)
PICK_SELF_CLEAR_MIN_MM = 8.0      # fingers must clear the part's OWN top material by this; the recorded opening
                                  # (tuned for the standing part) gave flat yellow 6.7 mm, and with the camera's
                                  # 2-10 mm pose error a finger came down on the bridge -> PROTECTIVE_STOP, twice


def _finger_clearance_mm(obj, pose, gid, open_mm, obstacles):
    """Smallest distance (mm) from either open-finger footprint (T along the closing axis, W across) to the
    part's own top-face material (CAD mesh at `pose`) and to each obstacle outline."""
    from spacing_check import FINGER_T_M, FINGER_W_M, GRASPS, gap_mm, finger_outlines
    from grasp_depth import load_mesh, surface_points
    from scipy.spatial.transform import Rotation as _R
    (p, q) = pose
    Rq = _R.from_quat(q)
    V, F = load_mesh(obj)
    P = Rq.apply(surface_points(V, F)) + np.asarray(p)
    mat = P[P[:, 2] > P[:, 2].max() - 0.004][:, :2]
    g = next(g for g in json.load(open(str(GRASPS).format(obj)))['grasp_points'] if g['id'] == gid)
    c = (Rq.apply([g['position'][k] for k in 'xyz']) + np.asarray(p))[:2]
    ax = Rq.apply([1.0, 0.0, 0.0])[:2]
    if np.linalg.norm(ax) < 0.5:
        ax = Rq.apply([0.0, 0.0, 1.0])[:2]
    ax = ax / np.linalg.norm(ax); pe = np.array([-ax[1], ax[0]])
    own = 1e9
    for sgn in (-1, 1):
        m = c + sgn * ax * (open_mm / 2000 + FINGER_T_M / 2)
        d = mat - m; u = d @ ax; v = d @ pe
        du = np.maximum(np.abs(u) - FINGER_T_M / 2, 0); dv = np.maximum(np.abs(v) - FINGER_W_M / 2, 0)
        own = min(own, float(np.sqrt(du ** 2 + dv ** 2).min()) * 1000)
    other = min([min(gap_mm(f, o) for f in finger_outlines(obj, pose, gid, open_mm)) for _, o in obstacles] or [1e9])
    return own, other


def _pick_room(obj, gid, width_mm, log_fh=None, base_pose=None) -> bool:
    """True when `obj` can be picked with grasp `gid` where it lies — choosing the pre-grasp opening for
    the stance it lies in (PICK_OPENING_MM): the recorded one or wider, whichever keeps the open fingers
    farthest from the part's own material and from base1 / the other parts. If no opening keeps them
    PICK_FINGER_CLEAR_MM from a neighbour, relocate the part first (disassemble --relocate) and return True
    on success; False only when the part is not seen or the relocation fails."""
    try:
        from spacing_check import observe_parts, PARTS, base_outline, outline
        # Speed T1 (2026-10-21): every part AND every other base from one still window; the run's base from its
        # applied read (it was re-observed by markers per pick: 11 reads, 64 s in one FMB1 run) and the layout's
        # parked region as an obstacle (base2/3/4 were probed one by one, 25 s)
        others = [b for b in ('base1', 'base2', 'base3', 'base4') if b != RUN_BASE['name']]
        seen, seen_bases = observe_parts(list(PARTS) + others, with_raw=True)
        if obj not in seen:
            # not in this view: its latest still sighting from any view (the check was skipped for line_green, picked
            # with 5 mm per side -> finger on the bar, 87 N, 2026-10-21)
            try:
                e_ = json.load(open(TRACKER_JSON)).get(obj)
                if e_ and time.time() - float(e_['t']) < TRACKER_TRUST_S:
                    seen[obj] = (tuple(e_['xyz']), tuple(e_['quat_xyzw']))
                    print(f"  [pick] {obj}: not in this view - judged at its tracked pose "
                          f"({1000*e_['xyz'][0]:.0f}, {1000*e_['xyz'][1]:.0f}) mm", flush=True)
            except Exception:
                pass
        ob = base_pose or RUN_BASE.get('pose') or observe_base_pose(RUN_BASE['name'])
        if obj not in seen or ob is None:
            return True                                   # cannot judge here; the pick's own checks apply
        obst = [(RUN_BASE['name'], base_outline(ob, RUN_BASE['name']))] + [(n, np.asarray(outline(n, seen[n]))[:, :2]) for n in seen if n != obj and n in PARTS]
        for other in others:                                      # another assembly's base on the table is an obstacle too
            if other in seen_bases:
                try:
                    obst.append((other, base_outline(seen_bases[other], other)))
                except Exception:
                    pass
        # (the layout's parked-region rectangle is NOT an obstacle here: it is a padded box - edge at x 0.22 with base3's
        # real edge ~0.17 - and every part set down at the staging edge read 7 mm from it and was sent to relocation,
        # 2026-10-21; the other base's measured outline above is the obstacle)
        best = None
        for open_mm in np.arange(width_mm, PICK_OPEN_MAX_MM + 0.1, 4.0):
            own, other = _finger_clearance_mm(obj, seen[obj], gid, float(open_mm), obst)
            score = min(own, other)
            if best is None or score > best[0] + 0.5:
                best = (score, float(open_mm), own, other)
        score, open_mm, own, other = best
        (px_, py_, pz_), pq_ = seen[obj]
        rpy_ = R_.from_quat(pq_).as_euler('xyz', degrees=True)
        msg = (f"  [pick] {obj} grasp {gid}: opening {open_mm:g} mm (recorded {width_mm:g}) -> fingers "
               f"{own:.0f} mm from the part itself, {other:.0f} mm from base1/others "
               f"(pose used: ({1000*px_:.0f}, {1000*py_:.0f}, {1000*pz_:.0f}) mm rpy ({rpy_[0]:.0f}, {rpy_[1]:.0f}, {rpy_[2]:.0f}))")
        print(msg, flush=True)
        if log_fh: log_fh.write(msg + "\n")
        PICK_OPENING_MM[obj] = open_mm
        if own < PICK_SELF_CLEAR_MIN_MM:
            msg = f"  [pick] {obj}: only {own:.0f} mm from its own material at any opening - picking anyway (watch the guard)"
            print(msg, flush=True)
            if log_fh: log_fh.write(msg + "\n")
        if other >= PICK_FINGER_CLEAR_MM:
            return True
        gaps = {'base1/others': other}; worst = 'base1/others'
        msg = (f"  [pick] {obj} grasp {gid}: open fingers {gaps[worst]:.0f} mm from {worst} "
               f"(< {PICK_FINGER_CLEAR_MM:g}) -> relocating it first")
        print(msg, flush=True)
        if log_fh: log_fh.write(msg + "\n")
        in_staging = seen[obj][0][0] >= 0.19 and seen[obj][0][1] <= -0.22
        r = run_subprocess([sys.executable, '-u', str(REPO_ROOT / 'ablations/disassemble.py'), '--assembly-json', RUN_BASE.get('json', 'x'),
                            '--relocate', obj, '--grasp-id', 'auto'] + (['--spot-staging'] if in_staging else []),
                           label=f'relocate {obj} (pick room)', timeout_s=900, log_fh=log_fh)
        if r.get('result') == 'failure':
            return False
        LAST_SEEN_XY.pop(obj, None)                        # it moved: look for it again
        return True
    except Exception as e:
        print(f"  [pick] room check failed ({e}) -> picking as recorded", flush=True)
        return True


TRACKER_JSON = REPO_ROOT / 'ablations' / 'logs' / 'part_tracker.json'
STATE_REF = {}                      # the current AssemblyState, for helpers outside replay_object
TRACKER_MAX_AGE_S = 1800.0
TRACKER_TRUST_S = 900.0          # a sighting this recent is where the part is when no view here sees it
PART_VIEW_Z_FALLBACKS = (0.45, 0.38)


def tracked(obj, max_age_s=TRACKER_MAX_AGE_S):
    """(x, y, age_s) of `obj`'s latest still-camera sighting from ANY view (ablations/part_tracker.py), or None."""
    try:
        e = json.load(open(TRACKER_JSON)).get(obj)
        age = time.time() - float(e['t'])
        return (float(e['xyz'][0]), float(e['xyz'][1]), age) if age <= max_age_s else None
    except Exception:
        return None


def _part_view(obj, log_fh=None) -> bool:
    """Camera straight above `obj`'s last-seen position (centred, closer) — the staging view at 0.62 m
    sees the whole region but detected a standing yellow only intermittently (2026-10-19: seen by the
    derive step, then not for 20 s at the pick). Confirms the part is seen (20 s) before returning
    True; falls back to the staging view, then fails."""
    xy = LAST_SEEN_XY.get(obj)
    tr = tracked(obj)
    if tr is not None and (xy is None or np.hypot(tr[0] - xy[0], tr[1] - xy[1]) > 0.01):
        print(f"  [view] {obj}: last seen by the tracker at ({1000*tr[0]:.0f}, {1000*tr[1]:.0f}) mm "
              f"{tr[2]:.0f} s ago -> look there", flush=True)
        xy = (tr[0], tr[1])
    b_ = camera_brightness()
    vz = PART_VIEW_Z_LOW_LIGHT if (b_ is not None and 0 <= b_ < LOW_LIGHT_BRIGHTNESS) else PART_VIEW_Z
    if vz != PART_VIEW_Z:
        print(f"  [view] low light (brightness {b_:.0f}) -> view {obj} from {vz:.2f} m", flush=True)
    if xy is not None:
        # far out a top-down view at PART_VIEW_Z has no IK path (2026-10-21: line_green at 0.80 m, view z 0.52):
        # come lower until one is reachable - the pick itself happens lower still
        for vz_ in [vz] + [z_ for z_ in PART_VIEW_Z_FALLBACKS if z_ < vz]:
            r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                f"{xy[0]:.4f}", f"{xy[1]:.4f}", f"{vz_:.4f}", '--quat', '0.0', '1.0', '0.0', '0.0'],
                               label=f'view above {obj} at {vz_:.2f} m', timeout_s=90, log_fh=log_fh)
            if r.get('result') != 'failure':
                break
        if r.get('result') != 'failure':
            time.sleep(1.5)
            if _seen_here(obj, 20.0):
                return True
            msg = f"  [view] {obj}: not seen from above its last position {tuple(round(v, 3) for v in xy)} -> staging view"
            print(msg, flush=True)
            if log_fh: log_fh.write(msg + "\n")
    if _staging_view(log_fh) and _seen_here(obj, 20.0):
        return True
    # A recent still-camera sighting from any view is where the part is (parts do not move by themselves);
    # move_to_grasp looks at it again from its own hover before it descends.
    tr = tracked(obj, TRACKER_TRUST_S)
    if tr is not None:
        LAST_SEEN_XY[obj] = (tr[0], tr[1])
        msg = (f"  [view] {obj}: no view here sees it, the tracker saw it at ({1000*tr[0]:.0f}, {1000*tr[1]:.0f}) mm "
               f"{tr[2]:.0f} s ago -> picking from there (move_to_grasp looks again)")
        print(msg, flush=True)
        if log_fh: log_fh.write(msg + "\n")
        return True
    return False


def _home_view(log_fh=None):
    """Second camera view for verify_seated: move_home (gripper empty, arm already above safe height
    over the base — the same move the between-parts step and the cleanup make next)."""
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                       label='verify second view: move_home', timeout_s=60, log_fh=log_fh)
    if r.get('result') != 'failure':
        time.sleep(1.5)                   # let the localizer catch up after the move
        return True
    return False


def verify_seated(objs, say, second_view=None) -> bool:
    """queries/verify_assembly (--xy-only) for each seated part, from the base view (arm above the
    base). Updates the ledger. A part not detected passes only when the ledger has it verified AND
    its occluder is seated (verified) after it. Fails closed — 2026-10-17 an empty-gripper 'seat' of
    u_brown went unnoticed for a whole assembly.

    second_view: callable that moves the camera to another view (move_home) and returns True on
    success. Used once, for a part never detected from above the base and not legitimately covered:
    seated u_brown lies on its side right under the camera there, markers edge-on — 2026-10-18 it was
    missed by all 8 reads from above and seen 30/30 from home, and a good cycle stopped."""
    led = _ledger(); ok_all = True; moved = False
    for o in reversed(list(objs)):        # covering parts first: their ledger time must be newer
        prev_ = led.get(o) or {}
        if (os.environ.get('SEAMS_CERTIFY') == '1' and prev_.get('verified') and prev_.get('how') == 'execute_mate'
                and time.time() - prev_.get('t', 0) < 1800):
            # certified by ExecuteMate this run; every later certification re-reads its predecessors
            say(f"  [verify] {o}: certified by ExecuteMate {time.time() - prev_['t']:.0f} s ago -> OK (not re-read)")
            continue
        r = _read_part(o)
        if _not_detected(r):
            prev = led.get(o) or {}
            occ = [c for c in OCCLUDED_BY.get(o, ()) if (led.get(c) or {}).get('verified')
                   and (led.get(c) or {}).get('t', 0) > prev.get('t', 0)]
            if prev.get('verified') and occ:
                say(f"  [verify] {o}: not visible (under {occ[0]}) — verified before it was covered -> OK")
                continue
            if prev.get('verified') and prev.get('how') == 'depth seat check' and time.time() - prev.get('t', 0) < 1800:
                say(f"  [verify] {o}: markers not seen — depth seat check confirmed it ({time.time() - prev['t']:.0f} s ago) -> OK")
                continue
            if second_view is not None and not moved:
                say(f"  [verify] {o}: not detected from above the base — second look from home")
                moved = True
                if second_view():
                    r = _read_part(o)
        if r.get('result') == 'success' and not _not_detected(r):
            led[o] = dict(verified=True, t=time.time(), how='marker xy %.1f mm' % (1000 * r.get('position_error_judged_m', 0)))
            e_ = r.get('position_error_m')
            if isinstance(e_, dict) and 'x' in e_ and 'y' in e_:   # base-frame xy error: follow_seated_offset reads it
                led[o]['rel_err_mm'] = [1000 * float(e_['x']), 1000 * float(e_['y'])]
            say(f"  [verify] {o}: OK ({led[o]['how']}, yaw err {(r.get('orientation_error_deg') or {}).get('yaw')} deg)")
            continue
        prev = led.get(o) or {}
        if (not _not_detected(r) and prev.get('how') == 'depth seat check' and prev.get('verified')
                and time.time() - prev.get('t', 0) < 1800 and float(r.get('position_error_judged_m') or 1.0) < 0.005
                and r.get('orientation_error_deg') in (None, {}, 'None')):
            # the marker read put it in place (xy) but returned no orientation; depth confirmed the seat minutes ago
            # (cycle 5: line_brown depth +0.02 mm, verify 'xy error 0.0 mm, orientation None' stopped the run)
            say(f"  [verify] {o}: marker xy {1000*float(r.get('position_error_judged_m') or 0):.1f} mm, no orientation read - "
                f"depth seat check confirmed it -> OK")
            continue
        if _not_detected(r):
            say(f"\n!!! [verify] {o}: not detected (or only tilted, unusable pose estimates) and not verified "
                f"earlier — cannot confirm it is seated")
        else:
            say(f"\n!!! [verify] {o}: NOT in its assembly pose — xy error "
                f"{1000 * float(r.get('position_error_judged_m') or 0):.1f} mm, orientation {r.get('orientation_error_deg')}")
        led.setdefault(o, {})['verified'] = False
        ok_all = False
    json.dump(led, open(ASSEMBLY_STATE, 'w'), indent=1)
    return ok_all


FOLLOW_READ_Z = 0.30          # flange height for the close read of a seated reference part
FOLLOW_READ_S = 10.0


HELD_YAW_MAX_DEG = 6.0        # a larger difference is a misread or a wrong fold branch, not a grasp error
HELD_YAW_APPLY = False        # 2026-10-20: three corrected line_green inserts (+2.26, +1.68, +1.92 deg read) had the
                              # slot turn the TCP by +0.05, -1.96, -2.96 deg at entry vs <= 0.8 deg uncorrected -
                              # the turn made the seat worse twice. Read and log only until an A/B with an
                              # independent reference settles the sign/bias of the close-range read.
HELD_READ_S = 6.0             # one bounded read; the U parts and yellow are never seen while held (2026-10-20:
                              # 0x at the regrasp +20 mm, at 0.30 m, at seat +45 mm) - the tool camera cannot see
                              # their markers in the grasp; line_green reads fine at safe height
HELD_READ_ABOVE_SEAT_M = 0.045   # flange above the predicted seat for the pre-insert read (U rim contact is +31 mm)
HELD_READ_MIN = 3
HELD_YAW_MIN_DEG = 0.3


PRE_INSERT_HOVER_DZ = 0.10      # flange above the predicted seat TCP for the pre-insert base read
PRE_INSERT_MIN_MM, PRE_INSERT_MIN_DEG = 0.5, 0.2     # smaller changes: keep the previous read
PRE_INSERT_MAX_MM, PRE_INSERT_MAX_DEG = 15.0, 3.0    # larger: a misread, keep the previous read
PRE_INSERT_MAX_MM_ICP, PRE_INSERT_MAX_DEG_ICP = 30.0, 10.0   # with a CAD-fit read (its own rms/inlier gates)


def choose_base_read(bname, ob_marker, ob_depth=None, say=print, tag='base'):
    """The base pose per BASE_FROM: 'marker' (default), 'depth' (top-face outline), 'icp' (6-dof CAD fit to the
    depth cloud in the camera frame, base_icp.read_icp; 2026-10-20: 1.8 mm median spread over 11 views, and it
    uses every surface, so a held part or seated parts hiding some of the base do not bias it). 'icp' falls back
    to the outline, then the markers, when the fit is loose. Call with the arm still, at the read's view."""
    return _choose_base_read(bname, ob_marker, ob_depth, say, tag)      # saved where a read is APPLIED


def _choose_base_read(bname, ob_marker, ob_depth=None, say=print, tag='base'):
    mode = os.environ.get('BASE_FROM', 'marker')
    choose_base_read.source = 'marker'
    if mode == 'icp' and ob_marker is not None:
        try:
            from base_icp import read_icp
            oi, info = read_icp(bname, ob_marker)
        except Exception as e:
            oi, info = None, {'error': str(e)}
        say(f"  [{tag}] CAD fit: {json.dumps(info)}")
        if oi is not None:
            choose_base_read.source = 'icp'
            return oi
    if mode in ('depth', 'icp') and ob_depth is not None:
        choose_base_read.source = 'depth'
        return ob_depth
    return ob_marker


def pre_insert_base_read(obj, state, args, say, log_fh=None):
    """Base read before the insert with the part in hand (PRE_INSERT_READ: 'shifted' default, 'hover' the old
    read from the insert hover, 'off')."""
    mode = os.environ.get('PRE_INSERT_READ', 'shifted')
    if mode == 'off':
        return
    if mode == 'hover':
        return _pre_insert_base_read_hover(obj, state, args, say, log_fh)
    return _pre_insert_base_read_shifted(obj, state, args, say, log_fh)


def _pre_insert_base_read_shifted(obj, state, args, say, log_fh=None):
    """Operator 2026-10-20: "reading a pose before insertion to correct can be done even with a part in hand if
    you rotate the wrist". The held part is rigid with the camera, so it always fills the same image region;
    held_clear_read picks a top-down view at the standard height (flange yaw 0/+-90, shifted <= 0.3 m) that
    puts the whole base in the rest of the image. Read markers + depth outline there, go back to the exact
    start pose (the held-orientation chain the insert uses stays valid), then apply the read."""
    ob = getattr(args, 'observed_base', None)
    if ob is None or state.held_quat is None:
        return
    try:
        from held_clear_read import held_in_flange, score_views, camera_K, COVER_OK
        from base_depth import VIEW_FLANGE_Z, measure as _depth
        from primitives.shared.config import held_tool_offset
        bname = args_base_name(args)
        snap = snapshot_robot_state(); tcp = snap.get('tcp') or {}
        if not tcp.get('xyz'):
            return
        K = camera_K()
        Tfo = held_in_flange(obj, state.last_grasp_id or 1, state.held_quat, tcp['xyz'], tcp['quat_xyzw'],
                             float(np.asarray(held_tool_offset())[2]))
        # views clear even with the held part's stereo shadow first, then views clear of its colour footprint
        # only (the shadow margin is conservative: for a held bar it ruled out every view, cycle 4); each is
        # only kept if the depth fit there actually succeeds (below)
        views = [v for v in score_views(obj, bname, ob, Tfo, tcp['quat_xyzw'], K, VIEW_FLANGE_Z)
                 if v['covered'] <= COVER_OK and v['outside'] <= 0.01]
        views += [v for v in score_views(obj, bname, ob, Tfo, tcp['quat_xyzw'], K, VIEW_FLANGE_Z, shadow=False)
                  if v['covered'] <= COVER_OK and v['outside'] <= 0.01]
    except Exception as e:
        say(f"  [base@insert] {obj}: no clear view computed ({e}) - keeping the previous read"); return
    if not views:
        say(f"  [base@insert] {obj}: no view of the base clear of the held part - keeping the previous read"); return
    back = ['primitives.move_cartesian', '--mode', 'real', '--xyz', *[f"{x:.4f}" for x in tcp['xyz']],
            '--quat', *[f"{x:.6f}" for x in tcp['quat_xyzw']], '--min-z', '0.15']
    # Try views until one gives a depth read (2026-10-20: views predicted clear gave ~1200 depth points for a held
    # line - the D415's stereo shadow behind a part 0.25 m away blanks more than the colour image shows). Distinct
    # shift directions first, at most PRE_INSERT_VIEW_TRIES.
    tried, picked = [], []
    for v in views:
        dirn = (np.sign(v['shift_mm'][0]), np.sign(v['shift_mm'][1]), v['turn'])
        if dirn in tried:
            continue
        tried.append(dirn); picked.append(v)
        if len(picked) >= PRE_INSERT_VIEW_TRIES:
            break
    nb = None
    for v in picked:
        say(f"  [base@insert] {obj}: view with the part in hand: wrist {v['turn']:+.0f} deg, shift {v['shift_mm']} mm "
            f"(base covered {v['covered']:.0%})")
        r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                            *[f"{x:.4f}" for x in v['flange_xyz']], '--quat', *[f"{x:.6f}" for x in v['flange_quat']],
                            '--min-z', '0.15'], label=f'base@insert: clear view ({obj})', timeout_s=90, log_fh=log_fh)
        if r.get('result') != 'success':
            continue
        nb = observe_base_pose(bname, timeout_s=6.0)
        if nb is not None and os.environ.get('BASE_FROM', 'marker') in ('depth', 'icp'):
            nd, info = _depth(nb, bname)
            say(f"  [base@insert] {obj}: depth outline {json.dumps(info)}")
            nb = choose_base_read(bname, nb, nd, say, tag=f'base@insert] [{obj}')
            if choose_base_read.source == 'marker':
                # never apply an oblique marker read (cycle 1: 5.2 mm / 1.5 deg off, three failed attempts)
                say(f"  [base@insert] {obj}: no depth read from this view")
                nb = None
        if nb is not None:
            break
    r2 = run_subprocess(['python3', '-u', '-m', *back], label=f'base@insert: back ({obj})', timeout_s=90, log_fh=log_fh)
    if r2.get('result') != 'success':                       # once more before failing closed
        r2 = run_subprocess(['python3', '-u', '-m', *back], label=f'base@insert: back, retry ({obj})', timeout_s=90, log_fh=log_fh)
    if r2.get('result') != 'success':
        say(f"\n!!! base@insert: could not return to the start pose ({r2.get('error')})")
        raise RuntimeError('pre-insert view: no return')
    if nb is None:
        say(f"  [base@insert] {obj}: base not read from the clear view - keeping the previous read"); return
    (x0, y0, _), q0 = ob; (x1, y1, _), q1 = nb
    dmm = 1000 * float(np.hypot(x1 - x0, y1 - y0))
    ddeg = float(np.degrees(abs((R_.from_quat(q1) * R_.from_quat(q0).inv()).as_euler('xyz')[2])))
    # a CAD fit carries its own quality gates (rms, inliers); a hard attempt really does push the base ~10 mm / 4 deg
    # (cycle 4: fork_orange's re-read was thrown away as a 'misread' at 3.9 deg)
    mx_mm, mx_deg = (PRE_INSERT_MAX_MM_ICP, PRE_INSERT_MAX_DEG_ICP) if getattr(choose_base_read, 'source', '') == 'icp' \
        else (PRE_INSERT_MAX_MM, PRE_INSERT_MAX_DEG)
    if dmm > mx_mm or ddeg > mx_deg:
        say(f"  [base@insert] {obj}: read moved {dmm:.1f} mm / {ddeg:.2f} deg - treated as a misread, kept the previous"); return
    if dmm < PRE_INSERT_MIN_MM and ddeg < PRE_INSERT_MIN_DEG:
        say(f"  [base@insert] {obj}: base unchanged ({dmm:.1f} mm, {ddeg:.2f} deg)"); return
    args.observed_base = nb
    _save_base_pose(bname, nb)
    say(f"  [base@insert] {obj}: base moved {dmm:.1f} mm / {ddeg:.2f} deg -> the insert aims at the clear-view read "
        f"({1000*x1:.1f}, {1000*y1:.1f}) mm")


def _pre_insert_base_read_hover(obj, state, args, say, log_fh=None):
    """Re-read the base from the insert hover, seconds before contact (operator 2026-10-20: the base is not
    fixed and moves during a part's sequence; the read from before the pick is minutes old and from 0.56 m).
    Close and straight above is the best view (2026-10-20 surveys). Updates args.observed_base, which the
    insert's argv is built from. Any failure keeps the previous read."""
    ob = getattr(args, 'observed_base', None)
    if ob is None:
        return
    try:
        from disassemble import seat_grasp
        tcp = seat_grasp(obj, state.last_grasp_id or 1, ob, base_name=args_base_name(args))
    except Exception as e:
        say(f"  [base@insert] {obj}: no seat estimate ({e}) - keeping the previous base read"); return
    snap = snapshot_robot_state(); cur = (snap.get('tcp') or {}).get('xyz')
    if not cur:
        return
    z_h = max(float(cur[2]), float(tcp[2]) + PRE_INSERT_HOVER_DZ)
    for lab, z in (('across at height', max(float(cur[2]), z_h)), ('down to the hover', z_h)):
        r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                            f"{tcp[0]:.4f}", f"{tcp[1]:.4f}", f"{z:.4f}", '--keep-orientation'],
                           label=f'base@insert: {lab} ({obj})', timeout_s=90, log_fh=log_fh)
        if r.get('result') != 'success':
            say(f"  [base@insert] {obj}: move failed - keeping the previous base read"); return
    time.sleep(1.0)
    nb = observe_base_pose(args_base_name(args), timeout_s=6.0)
    if nb is None:
        say(f"  [base@insert] {obj}: base not read from the hover - keeping the previous read"); return
    (x0, y0, _), q0 = ob; (x1, y1, _), q1 = nb
    dmm = 1000 * float(np.hypot(x1 - x0, y1 - y0))
    ddeg = float(np.degrees(abs((R_.from_quat(q1) * R_.from_quat(q0).inv()).as_euler('xyz')[2])))
    if dmm > PRE_INSERT_MAX_MM or ddeg > PRE_INSERT_MAX_DEG:
        say(f"  [base@insert] {obj}: read moved {dmm:.1f} mm / {ddeg:.2f} deg - treated as a misread, kept the previous"); return
    if dmm < PRE_INSERT_MIN_MM and ddeg < PRE_INSERT_MIN_DEG:
        say(f"  [base@insert] {obj}: base unchanged ({dmm:.1f} mm, {ddeg:.2f} deg)"); return
    if os.environ.get('PRE_INSERT_BASE_APPLY', '0') != '1':
        # 2026-10-20 FMB3: from the hover the held part hides markers and the base fills the image - the read
        # jumped 4.2 mm / 1.0 deg (line_brown) and 8.4 mm (line_red, nothing had moved the base); a read from the
        # standard view while holding a bar was 8 mm off in y. Logged only; the insert aims at the last read
        # taken with an empty gripper from the standard view. PRE_INSERT_BASE_APPLY=1 restores applying it.
        say(f"  [base@insert] {obj}: hover read differs {dmm:.1f} mm / {ddeg:.2f} deg from the last clean read "
            f"({1000*x1:.1f}, {1000*y1:.1f}) mm - logged, not applied (held-part reads are biased)"); return
    args.observed_base = nb
    say(f"  [base@insert] {obj}: base moved {dmm:.1f} mm / {ddeg:.2f} deg since the last read -> the insert aims at the fresh read "
        f"({1000*x1:.1f}, {1000*y1:.1f}) mm")


def args_base_name(args):
    return getattr(args, 'base_name', None) or 'base1'


def held_yaw_calibration(obj, state, say, log_fh=None, observed_base=None, lift_m=None):
    """Before the insert, with the part held at safe height: read the held part's yaw with the camera
    and turn the wrist in place by the difference to the orientation the chain assumes
    (state.held_quat), so the part - not just the flange - faces its slot. Nothing measured the part's
    yaw in the fingers after a pick/regrasp; the prismatic VERIFY showed ori_dev up to 3.85 deg at the
    seat on 2026-10-19 and every insert visibly turned the part at contact (operator). Compared modulo
    the part's fold symmetry. Skips when the part is not seen or the difference is outside
    (HELD_YAW_MIN_DEG, HELD_YAW_MAX_DEG)."""
    if state.held_quat is None:
        return
    from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
    from utils.data_path_finder import get_symmetry_dir
    if lift_m is not None:
        # Right after the grasp, lifted a little: the localizer publishes a part only near its rest height
        # (ray-snap tolerance 50 mm) - held u_brown was seen 8x/5 s at the rim but 0x at 0.30 m above the
        # base or the clear area (2026-10-20). This is also the one read that measures the grasp error.
        r = run_subprocess(['python3', '-u', '-m', 'primitives.move_to_safe_height', '--mode', 'real',
                            '--lift-by', f"{lift_m:.3f}"], label=f'held-yaw: lift {obj} {1000*lift_m:.0f} mm for the read',
                           timeout_s=60, log_fh=log_fh)
        if r.get('result') != 'success':
            return
        time.sleep(1.0)
    elif observed_base is not None:
        # Pre-insert read, low above the seat: after rotate_object the part shows the camera the face it was
        # seen with mid-insert (u_brown 8x/5 s at flange 0.186 m, 0x at 0.30 m and 0x lifted 20 mm at the
        # regrasp, 2026-10-20). Two moves: over at the current height, then straight down to seat + HELD_READ_ABOVE_SEAT_M
        # (the part's bottom stays above the rim contact, which is 31 mm above the seat for the U parts).
        try:
            from disassemble import seat_grasp
            tcp = seat_grasp(obj, state.last_grasp_id or 1, observed_base)
        except Exception as e:
            say(f"  [held-yaw] {obj}: no seat pose for the read ({e})")
            return
        snap = snapshot_robot_state(); cur = (snap.get('tcp') or {}).get('xyz')
        if not cur:
            return
        z_read = float(tcp[2]) + HELD_READ_ABOVE_SEAT_M
        for lab, z in (('over the seat', max(cur[2], z_read)), ('down for the read', z_read)):
            r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                f"{tcp[0]:.4f}", f"{tcp[1]:.4f}", f"{z:.4f}", '--keep-orientation'],
                               label=f'held-yaw: {lab} ({obj})', timeout_s=90, log_fh=log_fh)
            if r.get('result') != 'success':
                return
        time.sleep(1.0)
    # a held part is detected at only ~1-2 Hz (u_brown: 8 samples in 5 s, 2026-10-19 21:35) and the
    # localizer pauses after the lift: wait up to HELD_READ_S for HELD_READ_MIN samples, fold-aware mean
    obs = observe_object_orientation(obj, HELD_READ_S, n_samples=HELD_READ_MIN)
    S = [x for x in (obs.get('samples') or []) if x.get('quat_xyzw')]
    if len(S) < HELD_READ_MIN:
        say(f"  [held-yaw] {obj}: seen {len(S)}x in {HELD_READ_S:.0f} s while held -> no yaw calibration")
        return
    from spacing_check import mean_quat_fold_aware
    Rm = R_.from_quat(mean_quat_fold_aware(obj, [x['quat_xyzw'] for x in S])); Rc = R_.from_quat(state.held_quat)
    try:
        fold = load_symmetry_data(obj, str(get_symmetry_dir()))
        cands = [R_.from_matrix(m) for m in equivalent_orientations(Rm.as_matrix(), fold)]
    except Exception:
        cands = [Rm]
    best = min(cands, key=lambda c: (c.inv() * Rc).magnitude())
    d = Rc * best.inv()                                   # world rotation taking measured -> chain
    rv = d.as_rotvec(); err_deg = -np.degrees(rv[2])      # measured yaw - chain yaw about world z
    say(f"  [held-yaw] {obj}: held part reads {err_deg:+.2f} deg from the chain's orientation "
        f"(tilt part {np.degrees(np.hypot(rv[0], rv[1])):.1f} deg)")
    if not (HELD_YAW_MIN_DEG <= abs(err_deg) <= HELD_YAW_MAX_DEG):
        return
    # the measured in-hand yaw goes to the insert's hover (a wrist turn here is undone by the hover's absolute yaw)
    state.held_yaw_offset = float(err_deg)
    say(f"  [held-yaw] {obj}: in-hand yaw {err_deg:+.2f} deg -> the insert's hover takes it out")
    if not HELD_YAW_APPLY:
        say(f"  [held-yaw] {obj}: read only (HELD_YAW_APPLY off) - no turn")
        return
    snap = snapshot_robot_state(); tcp = (snap.get('tcp') or {})
    if not tcp.get('xyz') or not tcp.get('quat_xyzw'):
        return
    turn = R_.from_euler('z', -err_deg, degrees=True)
    q_new = (turn * R_.from_quat(tcp['quat_xyzw'])).as_quat()
    x, y, z = tcp['xyz']
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                        f"{x:.4f}", f"{y:.4f}", f"{z:.4f}", '--quat', *[f"{v:.6f}" for v in q_new]],
                       label=f'held-yaw: turn the wrist {-err_deg:+.2f} deg in place', timeout_s=60, log_fh=log_fh)
    if r.get('result') == 'success':
        # the part now really is where the chain assumed: state.held_quat stays as it is
        say(f"  [held-yaw] {obj}: wrist turned {-err_deg:+.2f} deg; the held part now matches the chain")


def record_follow_reads(obj, say, observed_base=None, log_fh=None, part_at_staging=False):
    """Before `obj`'s pick (arm free, gripper empty): read each part it sits between (FOLLOWS_SEATED)
    from a hover FOLLOW_READ_Z above its CAD seat and keep its base-frame xy error in the ledger
    (`rel_err_mm`) for follow_seated_offset. Close range because the base view (0.56 m) is not good
    enough here: 2026-10-19 20:56 it read the U parts within 1 mm of CAD, line_green then sat 5.5 mm
    into their openings for 13 s and slid 6 mm across before dropping in; the disassembly's hover reads
    the same evening put u_brown 7 mm off. Without a base read or a part not seen, the verify_seated
    entry (base view) stays, else the insert-based fallback is used."""
    parts = FOLLOWS_SEATED.get(obj)
    if not parts:
        return
    led = _ledger()
    if observed_base is not None:
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        (bx, by, bz), bq = observed_base
        Rb = R_.from_quat(bq)
        for p in parts:
            sp, _ = load_seat_pose_in_base(RUN_BASE['name'], p)
            seat_w = np.array([bx, by, bz]) + Rb.apply(sp)
            r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                f"{seat_w[0]:.4f}", f"{seat_w[1]:.4f}", f"{FOLLOW_READ_Z:.2f}", '--quat', '0.0', '1.0', '0.0', '0.0'],
                               label=f'follow read: hover above {p}', timeout_s=60, log_fh=log_fh)
            if r.get('result') != 'success':
                break
            time.sleep(1.0)
            o = observe_base_pose(p, timeout_s=FOLLOW_READ_S)
            if o is None:
                say(f"  [follow] {p}: not seen from {FOLLOW_READ_Z:.2f} m above its seat")
                continue
            e = Rb.inv().apply(np.array(o[0]) - seat_w)
            led.setdefault(p, {})['rel_err_mm'] = [1000 * float(e[0]), 1000 * float(e[1])]
            say(f"  [follow] {p}: seen ({1000*e[0]:+.1f}, {1000*e[1]:+.1f}) mm from its CAD seat (base frame, close read)")
    for p in parts:
        if isinstance((led.get(p) or {}).get('rel_err_mm'), list):
            continue
        say(f"  [follow] {p}: no close read -> {obj} follows the insert-based offsets")
    json.dump(led, open(ASSEMBLY_STATE, 'w'), indent=1)
    if observed_base is not None:
        # the reads left the arm above the base; the pick needs the camera on the part again
        # (2026-10-19 21:xx: line_green at the clear area was out of view -> pick failed)
        if part_at_staging:
            run_subprocess(['python3', '-u', '-m', 'primitives.core.move_to_clear_area', '--mode', 'real'],
                           label='follow read: back above the clear area', timeout_s=60, log_fh=log_fh)
        else:
            run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                           label='follow read: back home', timeout_s=60, log_fh=log_fh)


def reread_base(bname, prev, log_fh=None):
    """Re-read the base from straight above (flange at base_depth.VIEW_FLANGE_Z, top-down,
    gripper empty), depth outline first, marker read as fallback; returns the new pose or `prev`.
    Called between parts, where the arm is already at safe height over the base after the last
    release: it is the same vertical exit, just higher (not from home: there the base top sits at
    the D415's minimum depth range and the fit got 0 points, 2026-10-15).

    The UR5e rides a Vention 7th-axis slider: an insert, a nudge or a pull-out pushes the arm and
    the carriage creeps — the base then sits mm away IN THE ROBOT'S FRAME although it never moved
    on the table (2026-10-15: +1.9 mm over one assembly, +3.6 mm over one disassembly, +8 mm in an
    hour). So every part aims at a base read taken after the previous part's last push."""
    from base_depth import VIEW_FLANGE_Z as _VZ, measure as _base_from_depth
    from primitives.shared.config import DEFAULT_BASE_POSITION as _DBP
    cx, cy = (prev[0][0], prev[0][1]) if prev is not None else (_DBP[0], _DBP[1])
    r_ = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                         f"{cx:.4f}", f"{cy:.4f}", f"{_VZ:.2f}"],
                        label='base re-read move', timeout_s=60, log_fh=log_fh)
    if r_.get('result') != 'success':
        return prev
    ob = observe_base_pose(bname)
    if ob is None:
        return prev
    time.sleep(1.0)                                            # settle before the depth frames
    ob_d, info = _base_from_depth(ob, bname)
    ob = choose_base_read(bname, ob, ob_d, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")), 'base re-read')
    msg = (f"  [base] re-read ({'depth' if ob_d is not None else 'marker'}): "
           f"({1000*ob[0][0]:.1f}, {1000*ob[0][1]:.1f}) mm, yaw {np.degrees(2*np.arctan2(ob[1][2], ob[1][3])):.2f} deg")
    if prev is not None:
        msg += f"; moved ({1000*(ob[0][0]-prev[0][0]):+.1f}, {1000*(ob[0][1]-prev[0][1]):+.1f}) mm since the last read"
    print(msg, flush=True)
    if log_fh: log_fh.write(msg + "\n")
    _save_base_pose(bname, ob)
    return ob


def derive_sequence_for(obj: str, recorded: list, base_name: str, timeout_s: float = 30.0):
    """Build the tool sequence from where the part is actually lying.

    Returns (sequence, note). Falls back to the recorded sequence, unchanged, if the
    part cannot be observed or the planner cannot answer — deriving is an improvement
    on the recording, never a new way to fail.
    """
    obs = observe_object_orientation(obj, timeout_s)
    if obs.get('error') or not obs.get('samples'):
        return recorded, f"observe failed ({obs.get('error', 'no samples')}) -> recorded sequence"
    LAST_SEEN_XY[obj] = tuple(obs['samples'][-1]['xyz'][:2])
    try:
        from scipy.spatial.transform import Rotation as _R
        from primitives.shared.config import DEFAULT_BASE_ORIENTATION
        from primitives.shared.fold_symmetry import load_symmetry_data
        from utils.data_path_finder import (find_assembly_json_by_base_name,
                                            get_assembly_data_dir, get_symmetry_dir)
        from compliant_insertion_studio.planning.sequence_planner import (
            build_tool_sequence, plan_passes, quat_to_matrix)

        R_start = quat_to_matrix(obs['samples'][-1]['quat_xyzw'])
        jf = find_assembly_json_by_base_name(base_name or 'base1', str(get_assembly_data_dir()))
        components = json.load(open(jf))['components']
        target = next(c for c in components if c['name'] == obj)['rotation']['quaternion']
        R_target = (quat_to_matrix(DEFAULT_BASE_ORIENTATION)
                    @ quat_to_matrix([target[k] for k in 'xyzw']))
        fold = load_symmetry_data(obj, str(get_symmetry_dir()))
        passes = plan_passes(R_start, R_target, fold)
        sequence = build_tool_sequence(recorded, passes)

        rpy = _R.from_matrix(R_start).as_euler('xyz', degrees=True)
        note = (f"observed rpy=({rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}) -> "
                f"{passes} pass{'es' if passes > 1 else ''}, {len(sequence)} steps "
                f"({'same as recorded' if sequence == recorded else 'RESHAPED from recorded'})")
        return sequence, note
    except Exception as e:
        return recorded, f"derive failed ({e}) -> recorded sequence"


def make_env() -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}" if 'PYTHONPATH' in env else str(REPO_ROOT)
    )
    return env


PROGRAM_DROPPED = ('goal rejected', 'robot_program_running=False', 'External control program stopped',
                   'Controller is not running')


def run_subprocess(argv: list[str], label: str, timeout_s: int = 600, log_fh=None) -> dict:
    """_run_subprocess_once, retried once after re-sending the external-control program when the failure was the
    program dropping (2026-10-20: "Connection to reverse interface dropped" after a good move, no safety event)."""
    r = _run_subprocess_once(argv, label, timeout_s, log_fh)
    if r.get('result') == 'failure' and any(k in json.dumps(r) for k in PROGRAM_DROPPED) and not _SAFETY_STOP:
        rr = subprocess.run(['python3', '-m', 'primitives.core.ensure_program'], capture_output=True, text=True,
                            timeout=60, env=make_env(), cwd=str(REPO_ROOT))
        msg = f"  [program] dropped during '{label}' -> re-sent ({'ok' if rr.returncode == 0 else 'FAILED'}), retrying once"
        print(msg, flush=True)
        if log_fh: log_fh.write(msg + "\n")
        if rr.returncode == 0:
            r = _run_subprocess_once(argv, label + ' (retry)', timeout_s, log_fh)
    return r


def _run_subprocess_once(argv: list[str], label: str, timeout_s: int = 600,
                         log_fh=None) -> dict:
    """Spawn argv as a subprocess. Stream stdout to console + log_fh live.

    Uses os.setsid so the child becomes its own process-group leader; this
    lets the SIGINT handler kill the whole tree (gripper subprocesses,
    rclpy spinners, etc.) on operator Ctrl-C instead of orphaning them.
    """
    global _ACTIVE_PROC
    header = f"\n>>> {label}\n    cmd: {' '.join(argv)}\n"
    print(header, end='', flush=True)
    if log_fh is not None:
        log_fh.write(header)
        log_fh.flush()

    proc = subprocess.Popen(
        argv, cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=make_env(),
        preexec_fn=os.setsid,  # new process group → SIGINT handler can killpg
    )
    _ACTIVE_PROC = proc
    captured: list[str] = []
    # Watchdog: the deadline used to be checked only when the child printed a line, so a
    # silent hang (move_to_grasp waiting on a trajectory a protective stop aborted) blocked
    # the run for 27 min (2026-10-14). SIGTERM the group at the deadline; readline then ends.
    import threading
    timed_out = threading.Event()

    def _expire():
        timed_out.set()
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.kill()
    watchdog = threading.Timer(timeout_s, _expire)
    watchdog.daemon = True
    watchdog.start()
    # Safety watcher: a protective stop aborts the trajectory but the primitive waits on it, so
    # the run used to sit out the 60 s watchdog and then command more motions on a stopped robot
    # (2026-10-15: line_green pick with an open finger on base1). Poll the dashboard at 1 Hz and
    # interrupt the step the moment the robot leaves NORMAL.
    done_ev = threading.Event()

    def _watch_safety():
        global _SAFETY_STOP
        misses = 0
        while not done_ev.wait(1.0):
            st = robot_safety_status()
            misses = misses + 1 if st is None else 0
            if misses >= 3:                    # robot unreachable 3 s: the link dropped (2026-10-15:
                st = 'UNREACHABLE'             # the wired link went down 7 s mid-release, twice)
            if st is not None and st not in ('NORMAL', 'REDUCED'):
                _SAFETY_STOP = st
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    pass
                return
    threading.Thread(target=_watch_safety, daemon=True).start()
    try:
        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)
            captured.append(line)
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()
            if _INTERRUPTED:
                break
        if timed_out.is_set():
            _ACTIVE_PROC = None
            msg = f"!!! subprocess timeout after {timeout_s}s — killed pg\n"
            print(msg, end='', flush=True)
            if log_fh is not None:
                log_fh.write(msg)
            return {"result": "failure",
                    "error": f"subprocess timeout after {timeout_s}s",
                    "label": label}
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            proc.kill()
    finally:
        watchdog.cancel()
        done_ev.set()
        _ACTIVE_PROC = None
    if _SAFETY_STOP:
        msg = f"!!! robot safety status {_SAFETY_STOP} during '{label}' — step interrupted, run halts\n"
        print(msg, end='', flush=True)
        if log_fh is not None:
            log_fh.write(msg)
        return {"result": "failure", "error": f"robot {_SAFETY_STOP}", "label": label}
    text = ''.join(captured)
    rj = extract_result_json(text)
    if rj is None:
        return {"result": "failure" if proc.returncode != 0 else "success",
                "error": "no __RESULT_JSON__ block in output" if proc.returncode != 0 else None,
                "label": label, "returncode": proc.returncode}
    rj.setdefault("label", label)
    return rj


# ---------------------------------------------------------------------------
# Held-quat chain — mirrors run_assembly_step.py
# ---------------------------------------------------------------------------

class AssemblyState:
    """Tracks the current held_quat between primitive subprocesses.

    rotate_object emits final_object_orientation.quat in its __RESULT_JSON__
    after rotating the held part — that becomes the next held_quat.
    """

    def __init__(self) -> None:
        self.held_quat: Optional[Tuple[float, float, float, float]] = None
        self.last_grasp_id: Optional[int] = None  # carried from move_to_grasp
        self.last_grasp_object: Optional[str] = None
        self.held_yaw_offset: float = 0.0         # measured in-hand yaw of the held part (held read), deg; 0 = square
        self.held_tcp_quat = None                 # TCP orientation when held_quat was last set (the chain's flange)

    def update_from_result(self, tool_name: str, result: dict) -> None:
        if tool_name == 'move_to_grasp':
            self.held_yaw_offset = 0.0           # a new grasp: its in-hand yaw is unknown until read
        if not isinstance(result, dict):
            return
        # rotate_object emits final_object_orientation.quat (preferred when present)
        foo = result.get("final_object_orientation")
        if isinstance(foo, dict):
            q = foo.get("quat") if isinstance(foo.get("quat"), dict) else foo
            if isinstance(q, dict) and all(k in q for k in ('x', 'y', 'z', 'w')):
                self.held_quat = (
                    float(q['x']), float(q['y']), float(q['z']), float(q['w'])
                )
                return
        # move_to_grasp --pad-center reports how deep it grasped (flange -> grasp point). Every
        # later step that predicts the held part (insert hover/seat/contacts) reads it from
        # GRASP_TOOL_OFFSET_M (config.held_tool_offset); the disassembly reuses the recorded value.
        if result.get("movement_type") == "move_to_object" and result.get("grasp_tool_offset_m") is None:
            os.environ.pop('GRASP_TOOL_OFFSET_M', None)      # never carry another grasp's depth
        if result.get("grasp_tool_offset_m") is not None:
            os.environ['GRASP_TOOL_OFFSET_M'] = f"{float(result['grasp_tool_offset_m']):.5f}"
            if result.get("object_name"):
                from disassemble import record_grasp_offset
                record_grasp_offset(result["object_name"], result["grasp_tool_offset_m"])
        # move_to_grasp emits current_object_orientation.quat
        coo = result.get("current_object_orientation")
        if isinstance(coo, dict):
            q = coo.get("quat") if isinstance(coo.get("quat"), dict) else coo
            if isinstance(q, dict) and all(k in q for k in ('x', 'y', 'z', 'w')):
                self.held_quat = (
                    float(q['x']), float(q['y']), float(q['z']), float(q['w'])
                )

    def quat_argv(self) -> list[str]:
        if self.held_quat is None:
            return []
        return ['--current-object-orientation', *[f"{v:.6f}" for v in self.held_quat]]


# ---------------------------------------------------------------------------
# Tool → subprocess argv
# ---------------------------------------------------------------------------

# line_green drops between the arms of BOTH seated U parts (CAD: u_brown at base y -47.5 mm,
# u_orange at +48.5, line_green centred between them), so its slot is the line through where the
# U parts really seated — their force search finds that (seat TCP - CAD-predicted TCP; same
# orientation, so it is the part's own shift). Across the logged cycles line_green went straight
# in only when the two U parts sat at the same x (diff 0.05, -0.49 mm) and failed whenever they
# differed (-2.8, +6.3, +3.6 mm): a 3.6 mm difference over 96 mm turns the slot 2.2 deg, the
# bar's ends ~2.7 mm off (2026-10-15). Aim line_green at the slot's x and yaw; its position
# ALONG the bar (y) is its own pocket's, not the U parts'.
FOLLOWS_SEATED = {'line_green': ('u_brown', 'u_orange')}
MAX_FOLLOW_MM = 8.0
MAX_FOLLOW_YAW_DEG = 2.0          # from the camera's read of the two seated parts (base frame)
MAX_FOLLOW_YAW_INSERT_DEG = 1.5   # fallback, from the U parts' insert seat offsets: those mix the part's
# position with how it sat in the fingers. 2026-10-19 cycle 3: inserts said u_brown x -5.5 / u_orange
# +2.1 mm -> -4.5 deg, the camera said -2.0 / +0.9 -> -1.7 deg; line_green (CAD: +-1.1 mm in each U
# opening) wedged 4-8 mm into the openings on all 6 attempts. In 11 earlier runs today it was turned
# -3.2..+0.7 deg and always ended within 0.9 deg of the base (the openings turn it back on attempt 1,
# Rz free) - so the real slot turn is small and a wrong large turn is what fails.
SEATED_OFFSETS = REPO_ROOT / 'ablations/logs/seated_offsets.json'   # {part: seat - predicted TCP, mm}
# Written after every successful insert, entry removed when disassemble.py pulls the part out:
# it describes the parts seated NOW, whichever process seated them (a line_green-only retry
# finds the U parts' offsets without any time window).


def record_seat_offset(obj: str, since: float) -> None:
    """Store `obj`'s seat offset from the insert log written since `since` (if the log has one)."""
    try:
        from fmb1_loop import insert_summary
        rows = [r for r in insert_summary(since) if r.get('object') == obj and r.get('outcome') == 'success'
                and r.get('seat_minus_pred_mm')]
        if not rows:
            return
        # Total offset vs the UNbiased prediction = measured (vs the biased aim) + the bias applied.
        bx, by = _APPLIED_BIAS.get(obj, (0.0, 0.0))
        m = list(rows[-1]['seat_minus_pred_mm'])
        tot = [round(m[0] + 1000 * bx, 2), round(m[1] + 1000 * by, 2)] + m[2:]
        d = json.load(open(SEATED_OFFSETS)) if SEATED_OFFSETS.exists() else {}
        d[obj] = dict(seat_minus_pred_mm=tot, t=time.strftime('%Y-%m-%d %H:%M:%S'))
        json.dump(d, open(SEATED_OFFSETS, 'w'), indent=1)
        h = json.load(open(SEAT_HISTORY)) if SEAT_HISTORY.exists() else {}
        h.setdefault(obj, []).append(tot[:2]); h[obj] = h[obj][-20:]
        json.dump(h, open(SEAT_HISTORY, 'w'), indent=1)
    except Exception as e:
        print(f"  [seated-offsets] {obj}: not recorded ({e})")


def forget_seat_offset(obj: str) -> None:
    try:
        d = json.load(open(SEATED_OFFSETS))
        if d.pop(obj, None) is not None:
            json.dump(d, open(SEATED_OFFSETS, 'w'), indent=1)
    except Exception:
        pass


SEAT_HISTORY = REPO_ROOT / 'ablations/logs/seat_offset_history.json'   # {part: [[dx, dy] mm, ...]}
AIM_BIAS_PARTS = ('u_brown', 'u_orange', 'inverted_u_yellow')   # line_green follows the U parts instead
AIM_BIAS_N = 5                   # median of the last N seat offsets of the part
AIM_BIAS_MIN = 3                 # ...once there are at least this many
AIM_BIAS_MAX_MM = 8.0
_APPLIED_BIAS = {}               # part -> (dx, dy) m applied to this run's insert aim


def aim_bias(obj: str):
    """(dx, dy) world metres to add to `obj`'s insert aim: the median of where its last inserts
    actually seated relative to the prediction (2026-10-17: u_orange seated -4..-7 mm in y on every
    run and today -9.5 mm, beyond the 8 mm spiral, twice). The prior refined from contact data —
    the spiral then searches around where the part really goes."""
    if obj not in AIM_BIAS_PARTS or os.environ.get('NO_AIM_BIAS'):
        return 0.0, 0.0
    try:
        h = json.load(open(SEAT_HISTORY)).get(obj, [])
    except Exception:
        h = []
    if len(h) < AIM_BIAS_MIN:
        return 0.0, 0.0
    m = np.clip(np.median(np.array(h[-AIM_BIAS_N:]), axis=0), -AIM_BIAS_MAX_MM, AIM_BIAS_MAX_MM)
    return float(m[0]) / 1000, float(m[1]) / 1000


FOLLOW_GENERAL_MAX_MM = 4.0       # bound on the shift toward the seated parts' measured positions
FOLLOW_GENERAL_MAX_AGE_S = 7200.0


def _overlapping_seated(obj):
    """Parts seated before `obj` (assembly order, CAD) whose seated xy footprint overlaps obj's: what it interlocks
    with. No part names - from the CAD meshes at their seat poses in the run's base."""
    try:
        from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        from grasp_depth import load_mesh
        def box(p):
            V, _ = load_mesh(p); sp, sq = load_seat_pose_in_base(RUN_BASE['name'], p)
            W = V @ R_.from_quat(sq).as_matrix().T + sp
            return W[:, :2].min(0), W[:, :2].max(0)
        lo, hi = box(obj)
        out = []
        for p in get_seated_so_far(RUN_BASE['name'], obj):
            l2, h2 = box(p)
            if (l2 < hi).all() and (lo < h2).all():
                out.append(p)
        return out
    except Exception:
        return []


def _follow_seated_general(obj, base_quat):
    """2026-10-20 (operator: seated parts inside the CAD tolerance band are more data): aim `obj` by the measured
    seat errors (base frame, camera, verify ledger 'rel_err_mm') of the parts it interlocks with - fork_yellow aimed by
    the base landed on top of the two lines (each ~3-4 mm off CAD) every attempt. Mean, bounded, no yaw."""
    parts = _overlapping_seated(obj)
    if not parts:
        return 0.0, 0.0, 0.0
    led = _ledger()
    errs = [np.array(led[p]['rel_err_mm'], dtype=float) for p in parts
            if isinstance((led.get(p) or {}).get('rel_err_mm'), list) and time.time() - led[p].get('t', 0) < FOLLOW_GENERAL_MAX_AGE_S]
    if len(errs) < len(parts):
        print(f"  [follow] {obj}: measured seat errors of {parts} not all recorded -> aim at the base", flush=True)
        return 0.0, 0.0, 0.0
    # each reference constrains only ACROSS its long axis (a line is continuous along it: its along-error is marker
    # bias, irrelevant to what sits across it)
    from seat_check import TOP_FACE
    E = np.full((len(errs), 2), np.nan)
    for i, (p, e) in enumerate(zip(parts, errs)):
        ax = 0 if TOP_FACE[(p, RUN_BASE['name'])].get('long_axis', 'y') == 'y' else 1
        if TOP_FACE[(p, RUN_BASE['name'])]['half_long_m'] >= 2 * TOP_FACE[(p, RUN_BASE['name'])]['half_cross_m']:
            E[i, ax] = e[ax]                     # elongated: across only
        else:
            E[i] = e                             # compact: both
    m = np.array([np.nanmean(E[:, k]) if np.isfinite(E[:, k]).any() else 0.0 for k in (0, 1)])
    m = np.clip(m, -FOLLOW_GENERAL_MAX_MM, FOLLOW_GENERAL_MAX_MM)
    dw = R_.from_quat(base_quat).as_matrix()[:2, :2] @ m
    print(f"  [follow] {obj}: interlocks with {parts} -> aim shifted ({m[0]:+.1f}, {m[1]:+.1f}) mm (base frame, mean of "
          f"their measured seat errors {[np.round(e, 1).tolist() for e in errs]})", flush=True)
    return float(dw[0]) / 1000, float(dw[1]) / 1000, 0.0


def follow_seated_offset(obj: str, base_quat) -> Tuple[float, float, float]:
    """(dx, dy) world metres to add to `obj`'s insert base position and a yaw (rad) to turn its
    base frame by about the base origin, from the seat offsets of the parts it sits between
    (SEATED_OFFSETS). Zeros when they are not all there."""
    parts = FOLLOWS_SEATED.get(obj)
    if not parts:
        return _follow_seated_general(obj, base_quat)
    try:
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        d = json.load(open(SEATED_OFFSETS)) if SEATED_OFFSETS.exists() else {}
        last = {k: v['seat_minus_pred_mm'] for k, v in d.items()}
        if not all(p in last for p in parts):
            print(f"  [follow] {obj}: seat offsets of {parts} not all recorded ({SEATED_OFFSETS.name}) "
                  f"-> aim at the base")
            return 0.0, 0.0, 0.0
        Rb = R_.from_quat(base_quat).as_matrix()[:2, :2]
        (pa, pb) = parts
        led = _ledger()
        cam = {p: (led.get(p) or {}).get('rel_err_mm') for p in parts}
        if all(isinstance(v, list) and len(v) == 2 for v in cam.values()):
            da, db = (np.array(cam[p], dtype=float) for p in parts)           # mm, base frame (camera)
            src, cap = 'camera', MAX_FOLLOW_YAW_DEG
        else:
            da, db = (Rb.T @ np.array(last[p][:2]) for p in parts)           # mm, base frame (inserts)
            src, cap = 'inserts', MAX_FOLLOW_YAW_INSERT_DEG
        ya = 1000 * load_seat_pose_in_base(RUN_BASE['name'], pa)[0][1]; yb = 1000 * load_seat_pose_in_base(RUN_BASE['name'], pb)[0][1]
        dx_b = float(np.clip((da[0] + db[0]) / 2, -MAX_FOLLOW_MM, MAX_FOLLOW_MM))
        # slot direction from pa to pb: (db_x - da_x, yb - ya) -> angle from base +y
        dyaw = float(np.clip(-np.arctan2(db[0] - da[0], yb - ya), -np.radians(cap), np.radians(cap)))
        dw = Rb @ np.array([dx_b, 0.0])
        print(f"  [follow] {obj}: slot x {dx_b:+.1f} mm, yaw {np.degrees(dyaw):+.2f} deg (base frame, from the {src}"
              f", cap {cap:g} deg) — " + ", ".join(f"{p} x {d[0]:+.1f} mm" for p, d in zip(parts, (da, db))), flush=True)
        return float(dw[0]) / 1000, float(dw[1]) / 1000, dyaw
    except Exception as e:
        print(f"  [follow] {obj}: {e} -> aim at the base")
        return 0.0, 0.0, 0.0


def _seam_call(args_, timeout_s):
    p = subprocess.run(['python3', '-u', '-m', 'seams.client', *args_], capture_output=True, text=True,
                       timeout=timeout_s, env=make_env(), cwd=str(REPO_ROOT))
    line = next((l for l in reversed(p.stdout.splitlines()) if l.startswith('__SEAM__ ')), None)
    return json.loads(line[len('__SEAM__ '):]) if line else {'error': (p.stderr.strip().splitlines() or ['no answer'])[-1]}


def insert_via_drive_mate(obj, base, argv, snap, say):
    """SEAMS_DRIVE=1 (M3a): the insert through /ur5e/drive_mate instead of translate_object directly. Same target and
    grasp the direct call would use, expressed in the seam's terms: target = the base pose the insert aims at (aim
    corrections already applied, the per-part seat z removed - the server adds it), held = the part in the flange
    frame (grasp id + grasp depth ride in it). Result mapped back to the replay's dict."""
    from seams.client import mate_for
    from seams.drive_mate_server import held_pose
    from primitives.shared.config import get_object_base_offset_m, held_tool_offset

    def after(flag, n):
        i = argv.index(flag); return [float(v) for v in argv[i + 1:i + 1 + n]]
    mate = mate_for(obj, base)
    # the flange the chained held orientation was produced with (not now: after a wrist turn the world orientation is
    # stale - the in-hand yaw grew +1.23 -> +11.15 -> +19.19 deg over three attempts, 2026-10-21)
    tq = getattr(STATE_REF.get('state'), 'held_tcp_quat', None) or (snap.get('tcp') or {}).get('quat_xyzw')
    if mate is None or tq is None:
        return {'result': 'failure', 'error': f'drive_mate: no mate for {obj} on {base}' if mate is None else 'no TCP'}
    pos = after('--final-base-pos', 3); q = after('--final-base-orientation', 4)
    pos[2] -= float(get_object_base_offset_m(obj)[2])
    gid = int(after('--grasp-id', 1)[0]); held_q = after('--current-object-orientation', 4)
    off_ = float(getattr(STATE_REF.get('state'), 'held_yaw_offset', 0.0) or 0.0)
    if off_:
        held_q = list((R_.from_euler('z', off_, degrees=True) * R_.from_quat(held_q)).as_quat())   # the measured part
    hp, hq = held_pose(obj, gid, held_q, tq, float(np.asarray(held_tool_offset())[2]))
    say(f"  [drive_mate] {mate}: target ({1000*pos[0]:.1f}, {1000*pos[1]:.1f}) mm, grasp {gid}")
    out = _seam_call(['drive', '--mate', mate, '--target', *map(str, pos + q), '--held', *map(str, hp + hq)], 700)
    say(f"  [drive_mate] {mate}: {json.dumps(out)}")
    ok = bool(out.get('seated'))
    return dict(out, result='success' if ok else 'failure', movement_type='insert', via='drive_mate',
                error='' if ok else (out.get('error') or out.get('refusal') or f"stage {out.get('stage_reached')}"))


def certify_via_execute_mate(obj, base, observed_base, say, log_fh=None):
    """SEAMS_CERTIFY=1 (M3a): after a part is done, fly the seat view and ask /ur5e/execute_mate. Log-only until FMB1
    proves it: the replay's own verification still decides."""
    from seams.client import mate_for
    import seat_check as _sc
    mate = mate_for(obj, base)
    if mate is None or observed_base is None:
        return None
    out = None
    for side in _sc.VIEW_SIDES:                     # vision does no motion: the planner tries views until it measures
        r = run_subprocess(_sc.view_pose(obj, observed_base, base, side), label=f'execute_mate: seat view ({obj}, {side})',
                           timeout_s=90, log_fh=log_fh)
        if r.get('result') == 'failure':
            continue
        out = _seam_call(['execute', '--mate', mate], 450)
        if out.get('refusal') != 'engagement_invalid':
            break
        say(f"  [execute_mate] {mate}: not measurable from the {side} view ({out.get('refusal_detail', '')}) - next view")
    if out is None:
        say(f"  [execute_mate] {mate}: no seat view reached - not asked"); return None
    if out.get('placed'):
        led = _ledger()
        led[obj] = dict(verified=True, t=time.time(), how='execute_mate',
                        depth_mm=None if out.get('engagement_depth_m') is None else 1000 * float(out['engagement_depth_m']))
        json.dump(led, open(ASSEMBLY_STATE, 'w'), indent=1)
    say(f"  [execute_mate] {mate}: placed={out.get('placed')} refusal={out.get('refusal')!r} "
        f"{out.get('refusal_detail', '')} depth={out.get('engagement_depth_m')}")
    return out


DEPTH_VIEW_ZS = (0.52, 0.45)         # flange heights for the pre-pick depth read (the D415 returns no depth < ~0.45 m)
DEPTH_PICK_MAX_MOVE_M = 0.010        # a fit this far from its marker seed is not trusted (in clutter - a bar in the base -
DEPTH_PICK_MIN_MATCHED = 0.30        # one slid 28 mm on 5% of the points; table fits: 2.4 mm, 53%, 2026-10-21)


def depth_pick_target(obj, gid, say=print, log_fh=None):
    """The grasp point to pick `obj` at, from its CAD fitted to depth in a view straight above it (operator
    2026-10-21: "rely on depth to find the issue and fix perception"). From 0.45 m a depth fit of line_green
    repeated to 0.2 mm and agreed with its high marker read (2 mm) and where it had been set down (3 mm); the
    marker read move_to_grasp took from its low hover (no depth there) was 9-10 mm off and a finger landed on the
    bar (87 N). Returns ((x, y, z) grasp point, object quat) or None (move_to_grasp reads markers as before)."""
    tr = tracked(obj) or (LAST_SEEN_XY.get(obj) and (*LAST_SEEN_XY[obj], 0.0))
    if not tr:
        say(f"  [depth-pick] {obj}: no last sighting to look above - markers as before"); return None
    ok = False
    for vz in DEPTH_VIEW_ZS:
        r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                            f"{tr[0]:.4f}", f"{tr[1]:.4f}", f"{vz:.4f}", '--quat', '0.0', '1.0', '0.0', '0.0',
                            '--min-z', '0.15'], label=f'depth-pick view above {obj} at {vz:.2f} m', timeout_s=90,
                           log_fh=log_fh)
        if r.get('result') != 'failure':
            ok = True; break
    if not ok:
        say(f"  [depth-pick] {obj}: no reachable view above it - markers as before"); return None
    try:
        from part_depth import read as _read
        pose, info = _read(obj)
    except Exception as e:
        pose, info = None, {'error': str(e)}
    say(f"  [depth-pick] {obj}: {json.dumps(info)}")
    if pose is None:
        return None
    if (np.hypot(*info.get('moved_mm', [0, 0])) > 1000 * DEPTH_PICK_MAX_MOVE_M
            or info.get('n', 0) < DEPTH_PICK_MIN_MATCHED * max(info.get('n_data', 1), 1)):
        say(f"  [depth-pick] {obj}: fit moved {info['moved_mm']} mm on {info.get('n')}/{info.get('n_data')} points "
            f"- not trusted, markers as before"); return None
    import json as _j
    from primitives.shared.config import _GRASP_POINTS_DIR
    g = next(g_ for g_ in _j.load(open(f'{_GRASP_POINTS_DIR}/{obj}_grasp_points.json'))['grasp_points'] if g_['id'] == gid)
    (p, q) = pose
    gp = np.asarray(p) + R_.from_quat(q).apply([g['position'][k] for k in 'xyz'])
    LAST_SEEN_XY[obj] = (float(p[0]), float(p[1]))
    say(f"  [depth-pick] {obj}: grasp {gid} at ({1000*gp[0]:.1f}, {1000*gp[1]:.1f}, {1000*gp[2]:.1f}) mm from the depth fit")
    return tuple(float(v) for v in gp), tuple(float(v) for v in q)


def build_argv(tool_name: str, kw: dict, state: AssemblyState,
               base_offset_xy: Optional[Tuple[float, float]] = None,
               observed_base=None) -> Tuple[list[str], int]:
    """Map (tool_name, kwargs) → (subprocess argv, timeout_s). All calls forced to mode=real."""
    py = ['python3', '-u', '-m']

    if tool_name == 'control_gripper':
        cmd = str(kw.get('command', ''))
        return py + ['primitives.control_gripper', cmd, '--mode', 'real'], 30

    if tool_name == 'move_to_grasp':
        gid = int(kw.get('grasp_id', 1))
        state.last_grasp_id = gid
        state.last_grasp_object = kw['object_name']
        return py + ['primitives.move_to_grasp',
                     '--object-name', kw['object_name'],
                     '--grasp-id', str(gid),
                     '--mode', 'real', '--pad-center'], 60

    if tool_name == 'move_to_safe_height':
        argv = py + ['primitives.move_to_safe_height', '--mode', 'real']
        if 'object_name' in kw and kw['object_name']:
            argv += ['--object-name', kw['object_name']]
        return argv, 30

    if tool_name == 'rotate_object':
        argv = py + ['primitives.rotate_object',
                     '--mode', 'real',
                     '--object-name', kw['object_name'],
                     '--base-name', kw['base_name']]
        argv += state.quat_argv()
        return argv, 60

    if tool_name == 'translate_object':
        action = kw.get('action', '')
        argv = py + ['primitives.translate_object',
                     '--mode', 'real',
                     '--object-name', kw['object_name']]
        if 'base_name' in kw and kw['base_name']:
            argv += ['--base-name', kw['base_name']]
        if action == 'insert':
            argv += ['--insert', '--use-default-base-position',
                     '--insertion-type', 'compliant']
            # grasp_id is required by the compliant_insert path. Chain from the
            # most recent move_to_grasp; fall back to 1 if not seen.
            gid = state.last_grasp_id if state.last_grasp_id is not None else 1
            argv += ['--grasp-id', str(gid)]
            argv += state.quat_argv()
            if observed_base is not None:
                # Camera-observed base (see observe_base_pose). Per-object offsets keep only
                # their z (seat depth); their xy were fudges for the hand-tuned default base.
                from primitives.shared.config import get_object_base_offset_m
                (ox, oy, oz), oq = observed_base
                dz = float(get_object_base_offset_m(kw['object_name'])[2])
                fx, fy, fyaw = follow_seated_offset(kw['object_name'], oq)
                ox, oy = ox + fx, oy + fy
                bx_, by_ = aim_bias(kw['object_name'])
                if bx_ or by_:
                    _APPLIED_BIAS[kw['object_name']] = (bx_, by_)
                    ox, oy = ox + bx_, oy + by_
                    print(f"  [aim] {kw['object_name']}: + seat-history bias ({1000*bx_:+.1f}, {1000*by_:+.1f}) mm", flush=True)
                if fyaw:                       # turn the base frame about its origin (line_green sits there)
                    oq = tuple((R_.from_euler('z', fyaw) * R_.from_quat(oq)).as_quat())
                dyaw_exp = float(os.environ.get('INSERT_YAW_OFFSET_DEG', '0') or 0)
                if dyaw_exp:
                    # Experiment hook (2026-10-20 entry-yaw sweep): turn the insert's base frame by dyaw about the
                    # part's SEAT point, so only the entry yaw changes, not the aim position.
                    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
                    sp, _ = load_seat_pose_in_base(RUN_BASE['name'], kw['object_name'])
                    Rb_ = R_.from_quat(oq); o_ = np.array([ox, oy, oz])
                    seat_w = o_ + Rb_.apply(sp)
                    Rd = R_.from_euler('z', dyaw_exp, degrees=True)
                    o_new = seat_w + Rd.apply(o_ - seat_w)
                    ox, oy = float(o_new[0]), float(o_new[1])
                    oq = tuple((Rd * Rb_).as_quat())
                    print(f"  [yaw-exp] {kw['object_name']}: insert yaw offset {dyaw_exp:+.2f} deg about the seat point", flush=True)
                argv += ['--final-base-pos', f"{ox:.6f}", f"{oy:.6f}", f"{oz + dz:.6f}",
                         '--final-base-orientation', *[f"{v:.6f}" for v in oq]]
                argv = [a for a in argv if a != '--use-default-base-position']
                return argv, 600
            if base_offset_xy is not None and (base_offset_xy[0] != 0.0 or base_offset_xy[1] != 0.0):
                # Override default base position with an xy offset. Pulls
                # DEFAULT_BASE_POSITION at runtime so calibration stays the
                # source of truth.
                from primitives.shared.config import DEFAULT_BASE_POSITION, DEFAULT_BASE_ORIENTATION
                bx = float(DEFAULT_BASE_POSITION[0]) + float(base_offset_xy[0])
                by = float(DEFAULT_BASE_POSITION[1]) + float(base_offset_xy[1])
                bz = float(DEFAULT_BASE_POSITION[2])
                argv += ['--final-base-pos', f"{bx:.6f}", f"{by:.6f}", f"{bz:.6f}"]
                argv += ['--final-base-orientation',
                         *[f"{v:.6f}" for v in DEFAULT_BASE_ORIENTATION]]
                # Strip the --use-default-base-position flag since we're
                # passing an explicit pose now (mutually conflicting otherwise).
                argv = [a for a in argv if a != '--use-default-base-position']
            return argv, 600
        if action == 'place_down':
            argv += ['--place-down']
            argv += state.quat_argv()
            return argv, 600
        raise RuntimeError(f"Unknown translate_object action {action!r}")

    if tool_name == 'move_home':
        return py + ['primitives.move_home', '--mode', 'real'], 30

    raise RuntimeError(f"Unknown tool {tool_name!r}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _record_snapshot(snap_fh, when: str, label: str, snap: dict) -> None:
    """Append a snapshot line to the joint-state log (JSONL)."""
    if snap_fh is None:
        return
    snap_fh.write(json.dumps({
        'ts_iso': _dt.datetime.now().isoformat(),
        'when': when,           # 'before' | 'after'
        'label': label,
        'snapshot': snap,
    }) + "\n")
    snap_fh.flush()


INSERT_MAX_RECOVERIES = 1
PRE_INSERT_VIEW_TRIES = 3         # clear views tried for a depth read with the part in hand
INSERT_OUTER_TRIES = 3            # insert attempts with a fresh base read before each (PRISMATIC_MAX_RETRIES=1)
os.environ.setdefault('PRISMATIC_MAX_RETRIES', '1')
os.environ.setdefault('PRISMATIC_HOLD_IF_WEDGED', '1')     # a wedged bar stays put: released and nudged (see below)
SEAT_CHECK_OBJECTS = ('line_green',)      # FMB1's; any bar of any assembly is checked too (_seat_checked, 2026-10-20)
BAR_RATIO = 4.0                           # top face this much longer than wide = a bar (line_green 5.4, U parts 3.3)
CROSS_END_DIFF_MM = 1.5                   # one end this much higher than the other ...
CROSS_EDGE_DIFF_MM = 0.8                  # ... and at that end one long edge this much higher: riding a channel lip
CROSS_NUDGE_AT_MM = 12.0                  # push this far in from the high end (line_brown 2026-10-20: -58 of 70)


def _seat_checked(obj) -> bool:
    if obj in SEAT_CHECK_OBJECTS or os.environ.get('SEAT_CHECK_ALL', '1') == '1':   # every part (2026-10-20)
        return True
    try:
        from seat_check import TOP_FACE
        f = TOP_FACE[(obj, RUN_BASE['name'])]
        return f['half_long_m'] >= BAR_RATIO * f['half_cross_m']
    except Exception:
        return False


def _is_bar(obj) -> bool:
    try:
        from seat_check import TOP_FACE
        f = TOP_FACE[(obj, RUN_BASE['name'])]
        return f['half_long_m'] >= BAR_RATIO * f['half_cross_m']
    except Exception:
        return False


WEDGE_MAX_DZ_MM = 7.0       # stalled at most this far above the seat ...
WEDGE_MAX_DXY_MM = 5.0      # ... and this close to the aim in x/y (cycle 3: 3.8-4.4 mm, 5 mm high = on the channel edge)


def _wedged_near_seat(obj, since):
    """The last insert's end state from its raw CSV (not the FSM label): {'dz_mm', 'dxy_mm'} if it ended still,
    within WEDGE_MAX_DZ_MM above and WEDGE_MAX_DXY_MM of its predicted seat, else None."""
    import csv as _csv, glob as _glob
    metas = [m for m in _glob.glob(str(REPO_ROOT / f'compliant_insertion_studio/logs/insert_{obj}_*.meta.json'))
             if os.path.getmtime(m) >= since - 1]
    if not metas:
        return None
    mp = max(metas, key=os.path.getmtime)
    try:
        m = json.load(open(mp))
        rows = [x for x in _csv.DictReader(open(mp.replace('.meta.json', '.csv'))) if x.get('tcp_z') not in ('', 'nan', None)]
        P = np.array([[float(x['tcp_x']), float(x['tcp_y']), float(x['tcp_z'])] for x in rows[-50:]])
        tgt = np.array(m['predicted_seat_xyz'])
    except Exception:
        return None
    end = P[-1]; dz = 1000 * (end[2] - tgt[2]); dxy = 1000 * float(np.hypot(*(end[:2] - tgt[:2])))
    if 0 < dz <= WEDGE_MAX_DZ_MM and dxy <= WEDGE_MAX_DXY_MM and 1000 * np.ptp(P[:, 2]) < 1.0:
        return {'dz_mm': dz, 'dxy_mm': dxy}
    return None


def cross_nudge_plan(res):
    """(toward, at_long_mm) for a bar whose high end rides a channel lip, from the depth check: that end high
    and, at it, the edge on the lip side high - push from that side toward the other, at the high end
    (line_brown 2026-10-20: -y end +1.2, -x edge +2 mm -> toward +x at -58 mm; seated after). None if not."""
    ends = res.get('end_heights_mm'); per = (res.get('cross') or {}).get('end_edge_heights_mm'); f = res.get('face') or {}
    if not ends or not per or not f:
        return None
    hi = 1 if ends[1] > ends[0] else 0
    if ends[hi] - ends[1 - hi] < CROSS_END_DIFF_MM or per[hi] is None or None in per[hi]:
        return None
    lo_edge, hi_edge = per[hi]
    if abs(hi_edge - lo_edge) < CROSS_EDGE_DIFF_MM:
        return None
    cross_ax = 'x' if f['long_axis'] == 'y' else 'y'
    toward = ('+' if lo_edge > hi_edge else '-') + cross_ax      # away from the high edge
    at = (1 if hi else -1) * (1000 * f['half_long_m'] - CROSS_NUDGE_AT_MM)
    return toward, at
RESEAT_MAX = 1
# Tilted line_green = one end resting on base1's top past the end of its pocket (the 140 mm bar
# has almost no axial play). Operator's fix, verified 2026-10-14: gripper closed, from safe height,
# lower beside the high end, push ALONG the bar toward the slot in small steps at <= 5 N (base1 is
# glued to the table), re-check depth after each step. Regrasps and pressing did not work.
NUDGE_MAX_FORCE_N = 4.0              # push until the bar is blocked (seated or stuck): operator's rule,
                                     # replaces fixed step distances that had to guess the end (2026-10-15)
NUDGE_PUSH_PAST_MM = 15.0            # travel cap (operator): finger face at most this far past the estimated end
NUDGE_TRIES = 3
NUDGE_SHIFT_MM = 3.0                 # axial shift that means an end rides past its pocket
NUDGE_END_DIFF_MM = 2.5              # one end this much higher than the other: nudge, not reseat
NUDGE_START_GAPS_MM = (10.0, 20.0, 30.0)   # finger block this far past the depth end; next on contact
NUDGE_PAD_Z_BASE_M = 0.037           # pad centre height (base frame): tips 22 mm, base top 17.5 mm
NUDGE_AXIAL_BIAS_MM = 7.0          # depth puts a raised end ~7 mm past the real one (edge smear on
                                   # the D415). Not a base-read error: with the depth base and 0 here,
                                   # step 1 missed the bar entirely (operator, 2026-10-15)


def nudge_into_slot(obj, ob, res, say, out_dir) -> bool:
    """Push a tilted bar's raised end along its axis into its slot (operator's method): closed
    fingers lowered beside the high end from safe height, then ONE guarded push toward the slot
    that stops at NUDGE_MAX_FORCE_N — the bar slides while it is free and the force rises once it
    is blocked (seated against the pocket end, or stuck). Depth re-check after every try; the next
    try starts from the new measurement. The base is re-fitted by every check (7th-axis creep)."""
    from seat_check import check
    from primitives.shared.config import RG2_TIP_CLOSED_M, RG2_PAD_HALF_M, SAFE_HEIGHT
    base = [ob]
    f = res.get('face') or {'cx': 0.0, 'cy': 0.0, 'z': NUDGE_PAD_Z_BASE_M, 'long_axis': 'y'}
    def world(yb):          # yb: along the part's long axis from its CAD top-face centre (line_green: base y)
        (bx, by, bz), bq = base[0]
        p = np.array([f['cx'], f['cy'] + yb, f['z']]) if f['long_axis'] == 'y' else np.array([f['cx'] + yb, f['cy'], f['z']])
        return np.array([bx, by, bz]) + R_.from_quat(bq).as_matrix() @ p
    fz = lambda p: p[2] + RG2_TIP_CLOSED_M - RG2_PAD_HALF_M          # flange z, closed, top-down
    half = 10.1                                                      # closed finger block half-width (mm)
    q = ['0.0', '1.0', '0.0', '0.0']
    X = lambda p, z: [f"{p[0]:.4f}", f"{p[1]:.4f}", f"{z:.4f}"]
    go = lambda argv, label, t=60: run_subprocess(['python3', '-u', '-m', *argv], label=label, timeout_s=t)
    for k in range(1, NUDGE_TRIES + 1):
        ends, ext = res.get('end_heights_mm'), (res.get('axial') or {}).get('extent_mm')
        if not ends or not ext:
            say(f"  [nudge] {obj}: no end heights / axial extent -> cannot nudge"); return False
        sgn = 1.0 if ends[1] > ends[0] else -1.0                     # +1: the +y (base) end is high
        raw_mm = ext[1] if sgn > 0 else ext[0]                       # depth's end: never short of the real one
        end_mm = raw_mm - sgn * NUDGE_AXIAL_BIAS_MM                  # corrected (depth smears a raised end out)
        push = world((end_mm - sgn * NUDGE_PUSH_PAST_MM + sgn * half) / 1000.0)
        say(f"  [nudge] try {k}: {'+y' if sgn > 0 else '-y'} end high, depth end {raw_mm:+.0f} mm (base) "
            f"-> push toward the slot until {NUDGE_MAX_FORCE_N:g} N (cap {NUDGE_PUSH_PAST_MM:g} mm past {end_mm:+.0f})")
        for argv, label in [
                (['primitives.control_gripper', 'close', '--mode', 'real'], 'nudge close'),
                (['primitives.move_to_safe_height', '--mode', 'real'], 'nudge safe height')]:
            if go(argv, label).get('result') == 'failure':
                say(f"\n!!! {label} failed"); return False
        # Lower in clear space past the RAW depth end (the correction above moved a -y end 7 mm onto
        # the bar, 2026-10-15). Fingers must never land on the part: on contact, lift, step 10 mm
        # further out, lower again.
        for gap in NUDGE_START_GAPS_MM:
            start = world((raw_mm + sgn * (half + gap)) / 1000.0)
            r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, SAFE_HEIGHT + 0.08), '--quat', *q],
                   'nudge over the start')
            if r.get('result') == 'failure':
                say(f"\n!!! nudge over the start failed: {r.get('error', '')}"); return False
            r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, fz(start)), '--quat', *q,
                    '--min-z', '0.15', '--max-force', f"{NUDGE_MAX_FORCE_N}", '--min-duration', '5'],
                   f'nudge lower beside the end ({gap:g} mm clear)')
            if not r.get('contact'):
                break
            say(f"  [nudge] try {k}: touched something lowering {gap:g} mm past the end ({r.get('error')}) "
                f"-> lift, further out")
            if go(['primitives.move_to_safe_height', '--mode', 'real'], 'nudge lift').get('result') == 'failure':
                return False
        else:
            say(f"\n!!! {obj}: no clear spot past the end to lower the fingers"); return False
        if r.get('result') == 'failure':
            say(f"\n!!! nudge lower beside the end failed: {r.get('error', '')}"); return False
        time.sleep(1.5)                                              # settle before the force baseline
        r = go(['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(push, fz(push)), '--quat', *q,
                '--min-z', '0.15', '--max-force', f"{NUDGE_MAX_FORCE_N}", '--min-duration', '10'],
               f'nudge push {k} (until {NUDGE_MAX_FORCE_N:g} N)')
        say(f"  [nudge] push {k}: {'stopped at ' + str(r.get('error')) if r.get('contact') else 'reached the travel cap'}")
        for argv, label in [
                (['primitives.move_cartesian', '--mode', 'real', '--xyz', *X(start, fz(start) + 0.02), '--quat', *q,
                  '--min-z', '0.15'], 'nudge back off'),
                (['primitives.move_to_safe_height', '--mode', 'real'], 'nudge safe height')]:
            if go(argv, label).get('result') == 'failure':
                say(f"\n!!! {label} failed"); return False
        res = check(obj, base[0], move=True, out_dir=out_dir, base=RUN_BASE['name'])
        base[0] = res.get('observed_base') or base[0]
        say(f"  [seat-check] {obj} after nudge {k}: {json.dumps({k_: v for k_, v in res.items() if k_ not in ('profile_mm', 'observed_base')})}")
        if res.get('ok'):
            return True
        if _level_but_high(res) or abs(res.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG:
            say(f"  [nudge] {obj}: level along it now but high/rolled - stop nudging, edge press next")
            return False
    return False


EDGE_OFFSET_MM = 8.0                 # tap/press point off the bar's centreline (26 mm bar, 20 mm block)
EDGE_TAP_N = 3.0                     # measuring tap: a 3 N centre tap did not move the wedged bar
EDGE_PRESS_N = 5.0                   # operator's limit for pressing parts in the base
EDGE_DIFF_MIN_MM = 0.8               # edges closer than this: no roll to fix -> do not guess
ROLL_PRESS_DEG = 5.0                 # depth roll this large: a cocked bar -> edge press (a seated bar reads ~-3 deg: bias)


def _level_but_high(res) -> bool:
    """Depth says: level along the bar, centred along its axis, but the whole top is high —
    the bar is cocked across its width (roll), which depth cannot measure (16 mm strip; it reads
    -4 deg on every seat, seated or not: 54/54 checks, 2026-10-17)."""
    if 'pitch_deg' not in res or res.get('ok'):
        return False
    shift = abs(float((res.get('axial') or {}).get('shift_mm', 0.0)))
    return (abs(res['pitch_deg']) <= res.get('limits', {}).get('pitch_deg', 1.5) and shift < NUDGE_SHIFT_MM
            and res.get('height_err_mm', 0.0) > res.get('limits', {}).get('height_mm', 2.5))


def press_high_edge(obj, ob, say, out_dir) -> bool:
    """Rare roll wedge (operator, 2026-10-17: a bar cocked across its width sat 3.6 mm high by touch;
    a 5 N press on its high edge, off-centre at the grasp point, seated it). Find the high edge
    MECHANICALLY — tap both edges at 3 N — press only that edge at <= 5 N, re-check. No guessing:
    if the edges read the same there is no roll and it stops."""
    from tap_check import tap_once
    from seat_check import check
    from seat_check import TOP_FACE as _TF
    across_x = _TF[(obj, RUN_BASE['name'])].get('long_axis', 'y') == 'y'          # edges lie across the long axis
    off = lambda s: dict(offset_x_mm=s * EDGE_OFFSET_MM) if across_x else dict(offset_y_mm=s * EDGE_OFFSET_MM)
    taps = {s: tap_once(obj, ob, **off(s), force=EDGE_TAP_N, say=say, base=RUN_BASE['name']) for s in (-1, 1)}
    if not all(t['contact'] and t['err_mm'] is not None for t in taps.values()):
        say(f"  [edge] {obj}: an edge tap found no contact {taps} -> not pressing"); return False
    hi = max(taps, key=lambda s: taps[s]['err_mm'])
    diff = taps[hi]['err_mm'] - taps[-hi]['err_mm']
    if diff < EDGE_DIFF_MIN_MM:
        say(f"  [edge] {obj}: edges level ({taps[-1]['err_mm']:+.1f} / {taps[1]['err_mm']:+.1f} mm) — "
            f"not a roll, not pressing"); return False
    say(f"  [edge] {obj}: {'+x' if hi > 0 else '-x'} edge high by {diff:.1f} mm -> press it at {EDGE_PRESS_N:g} N")
    tap_once(obj, ob, **off(hi), force=EDGE_PRESS_N, press_mm=4.0, say=say, base=RUN_BASE['name'])
    res = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
    say(f"  [seat-check] {obj} after the edge press: "
        f"{json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}")
    return bool(res.get('ok'))


def seat_check_and_reseat(obj, state, args, say) -> bool:
    """The depth seat check (+ nudges / edge press); a pass is recorded in the verify ledger as a depth
    confirmation, which verify_seated accepts when the markers cannot see the part (2026-10-20: fork_orange,
    depth -0.08 mm from its CAD height, failed the run on 'marker not detected')."""
    ok = _seat_check_and_reseat(obj, state, args, say)
    if ok:
        led = _ledger()
        led[obj] = dict(verified=True, t=time.time(), how='depth seat check')
        json.dump(led, open(ASSEMBLY_STATE, 'w'), indent=1)
    return ok


def _seat_check_and_reseat(obj, state, args, say) -> bool:
    """Depth-check the seated part; if it sits tilted or high, take it with the insert's own
    grasp (the pose the insert released it from), and insert it again. Fails closed."""
    from seat_check import check
    from disassemble import seat_grasp, SEAT_EE_QUAT, HOVER_DZ
    ob = getattr(args, 'observed_base', None)
    if ob is None:
        say(f"  [seat-check] {obj}: no observed base -> skipped"); return True
    out_dir = REPO_ROOT / 'ablations' / 'logs' / 'seat_checks'
    for attempt in range(RESEAT_MAX + 1):
        res = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
        ob = res.get('observed_base') or ob                    # re-fitted from the check's own view
        args.observed_base = ob
        say(f"  [seat-check] {obj}: {json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}")
        if 'pitch_deg' in res and not res.get('ok') and (_level_but_high(res) or abs(res.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG
                                                           and abs(res['pitch_deg']) <= res.get('limits', {}).get('pitch_deg', 1.5)):
            # cocked across its width (operator 2026-10-20: "this is a roll issue ... we already faced a roll solve for
            # fmb1"): the FMB1 edge press - tap both edges, press the high one - not axial nudges
            say(f"  [seat-check] {obj}: level along it but {res.get('height_err_mm', 0):+.1f} mm high, roll {res.get('roll_deg', 0):+.1f} deg "
                f"-> edge press (FMB1 roll fix)")
            if press_high_edge(obj, ob, say, out_dir):
                return True
            if attempt >= RESEAT_MAX:
                say(f"\n!!! {obj} still not seated after the edge press"); return False
            continue
        plan = cross_nudge_plan(res) if 'pitch_deg' in res else None
        if abs(float((res.get('axial') or {}).get('shift_mm', 0.0))) >= NUDGE_SHIFT_MM:
            plan = None        # shifted along its length: the axial nudge first (line_red 2026-10-20: -16 mm,
                               # the sideways taps did nothing, the axial push seated it)
        if plan is not None and attempt < RESEAT_MAX:            # nudge, then the next pass re-checks
            # an end riding a channel lip: the depth limits call it ok (line_brown 2026-10-20, the operator
            # saw it unseated) - push that end sideways off the lip, to the CAD face (the base is not fixed)
            toward, at = plan
            say(f"  [seat-check] {obj}: high end on a channel lip -> sideways nudge toward {toward} at {at:+.0f} mm")
            r = run_subprocess(['python3', '-u', 'ablations/nudge_part.py', '--object', obj, '--base', RUN_BASE['name'],
                                f'--toward={toward}', '--at-long', f"{at:.0f}"], label=f'nudge {obj} {toward}', timeout_s=240)
            ob = observe_base_pose(RUN_BASE['name']) or ob
            args.observed_base = ob
            continue
        if res.get('ok'):
            return True
        if res.get('mode', '').startswith('median height'):
            if res.get('ok'):
                return True
            say(f"\n!!! {obj}: top {res.get('height_err_mm'):+.1f} mm from its CAD height - not seated"); return False
        if 'pitch_deg' not in res:
            say(f"  [seat-check] {obj}: could not measure -> continuing"); return True
        shift = abs(float((res.get('axial') or {}).get('shift_mm', 0.0)))
        ends = res.get('end_heights_mm') or [0.0, 0.0]
        if (abs(res['pitch_deg']) > res.get('limits', {}).get('pitch_deg', 1.5)
                or shift >= NUDGE_SHIFT_MM or abs(ends[1] - ends[0]) >= NUDGE_END_DIFF_MM):
            # tilted, or shifted along its axis = an end on base1's top past the pocket end; a
            # regrasp holds it where it is. 2026-10-15: a bar 4-7 mm high and +5 mm along its axis
            # (pitch only 1.2 deg) went to the top-down reseat, which cannot move it, and failed.
            ok = False
            shift = float((res.get('axial') or {}).get('shift_mm', 0.0))
            f_ = res.get('face') or {}
            if abs(shift) >= NUDGE_SHIFT_MM and f_:
                # taps along the bar to its CAD end face (operator 2026-10-20: a slow push drags the unfixed
                # base - cycle 1's axial push moved base3 10 mm); start clear of the MEASURED end
                lax = f_.get('long_axis', 'y')
                # direction from the HIGH end toward the slot (as nudge_into_slot): the extent midpoint is smeared
                # by a raised end (cycle 2: -y end 6.7 mm high, extent midpoint +7 -> taps pushed it further -y)
                ends_ = res.get('end_heights_mm') or [0.0, 0.0]
                toward = ('+' if ends_[0] > ends_[1] else '-') + lax
                say(f"  [seat-check] {obj}: shifted {shift:+.0f} mm along its length -> taps toward {toward} to the CAD end face")
                run_subprocess(['python3', '-u', 'ablations/nudge_part.py', '--object', obj, '--base', RUN_BASE['name'],
                                f'--toward={toward}', '--face-offset-mm', f"{abs(shift) + 3:.0f}", '--taps', '3'],
                               label=f'tap {obj} {toward}', timeout_s=300)
                res2 = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
                ob = res2.get('observed_base') or ob
                say(f"  [seat-check] {obj} after the taps: {json.dumps({k: v for k, v in res2.items() if k not in ('profile_mm', 'observed_base')})}")
                ok = bool(res2.get('ok'))
                res = res2 if 'pitch_deg' in res2 else res
            if not ok:
                ok = nudge_into_slot(obj, ob, res, say, out_dir)      # the slow push as the fallback
            if not ok:
                res2 = check(obj, ob, move=True, out_dir=out_dir, base=RUN_BASE['name'])
                if _level_but_high(res2) or abs(res2.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG:   # level but high: a roll
                    say(f"  [seat-check] {obj}: level but {res2['height_err_mm']:+.1f} mm high after the nudge -> edge taps")
                    ok = press_high_edge(obj, ob, say, out_dir)
                if not ok:
                    say(f"\n!!! {obj} still not seated after the nudge")
            return ok
        if _level_but_high(res):
            say(f"  [seat-check] {obj}: level and centred but {res['height_err_mm']:+.1f} mm high -> edge taps")
            ok = press_high_edge(obj, ob, say, out_dir)
            if not ok:
                say(f"\n!!! {obj} still not seated after the edge press")
            return ok
        if attempt == RESEAT_MAX:
            say(f"\n!!! {obj} still not flat after {RESEAT_MAX} reseat(s)"); return False
        from primitives.shared.config import get_grasp_id_for_assembly, get_gripper_width_mm
        gid = get_grasp_id_for_assembly(obj); w = get_gripper_width_mm(obj, gid)
        tcp = seat_grasp(obj, gid, ob, base_name=RUN_BASE['name'])
        from disassemble import grasp_offset_for
        os.environ['GRASP_TOOL_OFFSET_M'] = f"{grasp_offset_for(obj):.5f}"   # the insert's grasp depth
        q = [str(v) for v in SEAT_EE_QUAT]
        held = (R_.from_quat(ob[1]) * R_.from_quat(_seat_quat(obj, RUN_BASE['name']))).as_quat()
        (bx, by, bz), bq = ob
        from primitives.shared.config import get_object_base_offset_m
        dz = float(get_object_base_offset_m(obj)[2])
        steps = [
            (['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{tcp[0]:.4f}", f"{tcp[1]:.4f}",
              f"{tcp[2] + HOVER_DZ:.4f}", '--quat', *q], 'reseat hover', 60),
            (['primitives.control_gripper', f"{w:g}", '--mode', 'real'], 'reseat open', 30),
            (['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{tcp[0]:.4f}", f"{tcp[1]:.4f}",
              f"{tcp[2]:.4f}", '--quat', *q, '--min-z', '0.15', '--max-force', '12', '--min-duration', '6'],
             'reseat descend', 60),
            (['primitives.control_gripper', 'close', '--mode', 'real'], 'reseat close', 30),
            (['primitives.translate_object', '--mode', 'real', '--object-name', obj, '--base-name', RUN_BASE['name'],
              '--insert', '--insertion-type', 'compliant', '--grasp-id', str(gid),
              '--current-object-orientation', *[f"{v:.6f}" for v in held],
              '--final-base-pos', f"{bx:.6f}", f"{by:.6f}", f"{bz + dz:.6f}",
              '--final-base-orientation', *[f"{v:.6f}" for v in bq]], 'reseat insert', 600),
            (['primitives.control_gripper', f"{w:g}", '--mode', 'real'], 'reseat release', 30),
            (['primitives.move_to_safe_height', '--mode', 'real'], 'reseat lift', 60),
        ]
        say(f"  [seat-check] {obj} not flat -> reseat {attempt + 1}/{RESEAT_MAX} with the insert grasp")
        for argv, label, t in steps:
            r = run_subprocess(['python3', '-u', '-m', *argv], label=label, timeout_s=t)
            if r.get('result') == 'failure':
                say(f"\n!!! {label} failed: {r.get('error', '')}"); return False
    return False


def _seat_quat(obj, base_name='base1'):
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    return load_seat_pose_in_base(base_name, obj)[1]


CAMERA_TOPIC = os.environ.get('CAMERA_HEALTH_TOPIC', '/camera/camera/color/image_raw')
CAMERA_MAX_AGE_S = 2.0


def camera_frame_age(timeout_s: float = 3.0):
    """Seconds since the newest colour frame, or None if none arrived within timeout_s.
    The D415 dropped off USB mid-run on 2026-10-14; the localizer kept publishing empty pose
    lists and move_to_grasp spent its retries on 'not detected' instead of saying why."""
    import rclpy
    from sensor_msgs.msg import Image
    own = not rclpy.ok()
    if own:
        rclpy.init()
    node = rclpy.create_node('replay_camera_check'); st = {}
    node.create_subscription(Image, CAMERA_TOPIC, lambda m: st.__setitem__('t', time.time()), 2)
    end = time.time() + timeout_s
    while 't' not in st and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_node()
    if own:
        rclpy.shutdown()
    return (time.time() - st['t']) if 't' in st else None


ROTATE_SKIP_DEG = 5.0


def already_at_target(obj, held_quat, base_name) -> bool:
    """True when the held part's chained orientation is within ROTATE_SKIP_DEG of a fold-symmetry
    equivalent of its assembly target (what rotate_object would compute and then not move)."""
    if held_quat is None:
        return False
    try:
        from scipy.spatial.transform import Rotation as _R
        from primitives.shared.config import DEFAULT_BASE_ORIENTATION
        from primitives.shared.fold_symmetry import load_symmetry_data, equivalent_orientations
        from utils.data_path_finder import (find_assembly_json_by_base_name, get_assembly_data_dir,
                                            get_symmetry_dir)
        from compliant_insertion_studio.planning.sequence_planner import quat_to_matrix
        jf = find_assembly_json_by_base_name(base_name or 'base1', str(get_assembly_data_dir()))
        target = next(c for c in json.load(open(jf))['components'] if c['name'] == obj)['rotation']['quaternion']
        R_target = quat_to_matrix(DEFAULT_BASE_ORIENTATION) @ quat_to_matrix([target[k] for k in 'xyzw'])
        R_held = _R.from_quat(held_quat).as_matrix()
        fold = load_symmetry_data(obj, str(get_symmetry_dir()))
        for R_eq in equivalent_orientations(R_target, fold):
            ang = np.degrees(_R.from_matrix(R_held.T @ np.asarray(R_eq)).magnitude())
            if ang < ROTATE_SKIP_DEG:
                return True
    except Exception as e:
        print(f"  [rotate-skip] check failed ({e}) -> rotating as recorded")
    return False


def replay_object(entry: dict, state: AssemblyState, args: argparse.Namespace,
                   log_fh=None, snap_fh=None) -> bool:
    obj = entry['object_name']
    sequence = entry.get('tool_sequence', []) or []

    print(f"\n{'='*72}\n  Replaying object: {obj}  ({len(sequence)} steps)\n{'='*72}")
    if not args.dry_run and not getattr(args, 'part_at_staging', False) or (not args.dry_run and state.held_quat is None):
        # gripper empty between parts: unwind wrist_3 if the day's moves wound it up (camera cable; 2026-10-20)
        run_subprocess(['python3', '-u', '-m', 'primitives.core.unwind_wrist', '--mode', 'real'],
                       label='unwind wrist_3 if wound', timeout_s=120, log_fh=log_fh)

    # Base the object is headed for — move_to_grasp uses it to pick the wrist whose
    # reorient keeps the camera away from the table (primitives/shared/camera_up.py).
    base_name = None
    for s_ in sequence:
        try:
            base_name = parse_tool_call(s_)[1].get('base_name') or base_name
        except Exception:
            pass

    def _move_down_floor_z(ee_quat_xyzw):
        """move_down's flange floor for this flange orientation (its --to-table rule when face-down)."""
        from primitives.shared.config import (TABLE_HEIGHT, RG2_TIP_CLOSED_M, GRIPPER_CENTER_TOOL_OFFSET,
                                              TABLE_COLLISION_MARGIN_SIDEWAYS)
        Rm = R_.from_quat(ee_quat_xyzw).as_matrix()
        if Rm[2, 2] < -0.5:      # face-down: closed fingertips 1 mm above the table
            return TABLE_HEIGHT + 0.001 - (Rm @ np.array([0.0, 0.0, RG2_TIP_CLOSED_M]))[2]
        return TABLE_HEIGHT + TABLE_COLLISION_MARGIN_SIDEWAYS - (Rm @ GRIPPER_CENTER_TOOL_OFFSET)[2]

    def _gentle_place_down(step_idx, tag):
        """translate_object --place-down with the disassembly's landing (disassemble.place_at): the same
        move_to_clear_area, then move_down to LAND_SLOW_ABOVE_M above the CAD touch height of the held
        stance, then a slow move_cartesian that stops at LAND_FORCE_N. move_down --to-table alone aims at
        the closed-tip floor, below a part held by its top: yellow held by the bridge (recovery set-down,
        2026-10-19) hit the table at 37 N, 60 N peak. Returns None (caller runs the plain primitive) when
        the touch height cannot be computed."""
        off = os.environ.get('GRASP_TOOL_OFFSET_M')
        gid = state.last_grasp_id
        if state.held_quat is None or off is None or gid is None:
            return None
        from disassemble import contact_flange_z, LAND_SLOW_ABOVE_M, LAND_SPEED_M_S, LAND_FORCE_N
        try:
            touch_gp_z = contact_flange_z(obj, gid, state.held_quat, 0.0)   # grasp-point z when the part touches
        except Exception as e:
            say(f"  [{step_idx}] place_down: no CAD touch height ({e}) - plain move_down")
            return None
        label = f"{obj}#{step_idx}{tag} place_down"
        py3 = ['python3', '-u', '-m']
        r = run_subprocess(py3 + ['primitives.core.move_to_clear_area', '--move', '--mode', 'real',
                                  '--object-name', obj], label=f"{label} (clear area)", timeout_s=45, log_fh=log_fh)
        if r.get('result') == 'failure':
            return dict(r, movement_type='place_down')
        snap = snapshot_robot_state()
        tcp = snap.get('tcp') or {}
        if not tcp.get('xyz') or not tcp.get('quat_xyzw'):
            say(f"  [{step_idx}] place_down: no TCP pose after the clear-area move - plain move_down")
            r = run_subprocess(py3 + ['primitives.core.move_down', '--mode', 'real', '--to-table'],
                               label=f"{label} (lower)", timeout_s=310, log_fh=log_fh)
            return dict(r, movement_type='place_down')
        x, y, _ = tcp['xyz']
        floor = _move_down_floor_z(tcp['quat_xyzw'])
        # grasp depth is along the tool z axis: vertical for a top-down gripper, flat for a sideways one
        cz = touch_gp_z - (R_.from_quat(tcp['quat_xyzw']).as_matrix() @ np.array([0.0, 0.0, float(off)]))[2]
        # Both stages stay at or above move_down's floor (a sideways stance predicts the touch below it;
        # the plain primitive stopped at the floor there too). Slow stage: <= 25 mm, ~6 s.
        z_slow = max(cz, floor) + LAND_SLOW_ABOVE_M
        z_end = max(floor, cz - 0.010)        # 10 mm past the predicted touch
        say(f"  [{step_idx}] place_down: CAD touch at flange z {cz:.4f} (floor {floor:.4f}); slow from "
            f"{z_slow:.4f} to {z_end:.4f} until {LAND_FORCE_N:g} N")
        if tcp['xyz'][2] > z_slow + 0.002:
            r = run_subprocess(py3 + ['primitives.core.move_down', '--mode', 'real', '--to-table',
                                      '--height', f"{z_slow:.4f}"],
                               label=f"{label} (lower to {1000*LAND_SLOW_ABOVE_M:.0f} mm above the touch)",
                               timeout_s=310, log_fh=log_fh)
            if r.get('result') == 'failure':
                return dict(r, movement_type='place_down')
        dur = max(3.0, (z_slow - z_end) / LAND_SPEED_M_S)
        r = run_subprocess(py3 + ['primitives.move_cartesian', '--mode', 'real', '--xyz', f"{x:.4f}", f"{y:.4f}",
                                  f"{z_end:.4f}", '--keep-orientation', '--min-z', f"{z_end - 0.001:.4f}",
                                  '--max-force', f"{LAND_FORCE_N:g}", '--min-duration', f"{dur:.1f}"],
                           label=f"{label} (land, slow, until {LAND_FORCE_N:g} N)", timeout_s=120, log_fh=log_fh)
        if r.get('result') == 'failure' and r.get('contact') and str(r.get('error', '')).startswith('contact'):
            say(f"  [{step_idx}] place_down: touched down ({r.get('error')})")
            r = dict(r, result='success')
        return dict(r, movement_type='place_down')

    def execute(step_idx, step, tool_name, kw, tag=""):
        """Run one primitive. Returns its result dict, or None if the argv build failed."""
        if tool_name == 'translate_object' and kw.get('action') == 'place_down':
            snap_before = snapshot_robot_state()
            _record_snapshot(snap_fh, "before", f"{obj}#{step_idx}{tag} place_down", snap_before)
            r = _gentle_place_down(step_idx, tag)
            if r is not None:
                state.update_from_result(tool_name, r)
                _record_snapshot(snap_fh, "after", f"{obj}#{step_idx}{tag} place_down", snapshot_robot_state())
                return r
        if tool_name == 'move_to_grasp':
            age = camera_frame_age()
            if age is None or age > CAMERA_MAX_AGE_S:
                msg = (f"camera not streaming (last colour frame {'never' if age is None else f'{age:.1f} s ago'})"
                       " - check the D415 USB link (dmesg), restart realsense2_camera + localizer")
                say(f"\n!!! {msg}")
                return {'result': 'failure', 'error': msg}
        try:
            argv, timeout_s = build_argv(tool_name, kw, state,
                                         base_offset_xy=args.base_offset_xy,
                                         observed_base=getattr(args, 'observed_base', None))
        except Exception as e:
            print(f"  [{step_idx}] BUILD FAILED ({e}); aborting.")
            return None
        if tool_name == 'move_to_grasp' and base_name:
            argv += ['--base-name', base_name]
        if tool_name == 'move_to_grasp' and grasp_width['mm'] is not None:
            # the part's FIRST pick (step 2 of every sequence) uses the stance-chosen opening; regrasps as recorded
            w_ = PICK_OPENING_MM.get(obj, grasp_width['mm']) if (tag == "" and step_idx == 2) else grasp_width['mm']
            # floor: >= PICK_POSE_MARGIN_MM of open finger on each side of the part (2026-10-21: line_green picked with
            # 5 mm per side, table reads 5-10 mm off -> a finger landed on the bar's top, 87 N, protective stop).
            # Command units: a commanded opening gives a tip gap ~PICK_CMD_TO_GAP_MM smaller (39.8 -> 35.8 measured).
            try:
                from primitives.shared.config import cad_grip_width_mm as _cgw
                floor_ = float(_cgw(obj, state.last_grasp_id or int(kw.get('grasp_id', 1)))) + 2 * PICK_POSE_MARGIN_MM + PICK_CMD_TO_GAP_MM
                # only where nothing else judged the opening, and only for a bar: a concave part (U, inverted U)
                # limits the opening by its own other leg, which the room check weighs (u_brown: 7 mm at 39)
                if w_ < floor_ and obj not in PICK_OPENING_MM and _is_bar(obj):
                    say(f"  [{step_idx}] pre-grasp {w_:g} mm -> {floor_:.1f} mm ({PICK_POSE_MARGIN_MM:g} mm of finger room per side)")
                    w_ = floor_
            except Exception:
                pass
            argv += ['--pre-grasp-width', f"{w_:g}"]
        if tool_name == 'move_to_grasp' and os.environ.get('PICK_FROM', 'depth') == 'depth' and not args.dry_run:
            tgt = depth_pick_target(obj, int(kw.get('grasp_id', state.last_grasp_id or 1)), say, log_fh)
            if tgt is not None:
                argv += ['--target-xyz', *[f"{v:.5f}" for v in tgt[0]], '--target-xyzw', *[f"{v:.6f}" for v in tgt[1]]]
        label = f"{obj}#{step_idx}{tag} {tool_name}"
        if tool_name == 'rotate_object' and already_at_target(obj, state.held_quat, base_name):
            # Pass 2 of a 2-pass part: the regrasp already holds it at the target orientation and
            # rotate_object moves 0.01 rad. It is NOT skippable, though (tried 2026-10-19: two rim
            # stalls in a row, 51 mm off): its OUTPUT quaternion is the snapped fold branch the
            # insert's CAD chain (predict_tcp_at_seat -> grasp offset direction) was validated with;
            # the camera's raw quaternion is the other fold branch of this symmetric part and
            # mirrors the leg grasp. Run it; only note the wasted step.
            say(f"  [{step_idx}] rotate_object: {obj} already at a fold-equivalent of its target "
                f"(< {ROTATE_SKIP_DEG:g} deg) - running it for its snapped orientation")
        snap_before = snapshot_robot_state()
        _record_snapshot(snap_fh, "before", label, snap_before)
        STATE_REF['state'] = state
        if tool_name == 'translate_object' and kw.get('action') == 'insert' and os.environ.get('SEAMS_DRIVE') == '1':
            result = insert_via_drive_mate(obj, base_name or RUN_BASE['name'], argv, snap_before, say)
        elif tool_name == 'translate_object' and kw.get('action') == 'insert' and getattr(state, 'held_yaw_offset', 0.0):
            os.environ['HELD_YAW_OFFSET_DEG'] = f"{state.held_yaw_offset:.3f}"
            try:
                result = run_subprocess(argv, label=label, timeout_s=timeout_s, log_fh=log_fh)
            finally:
                os.environ.pop('HELD_YAW_OFFSET_DEG', None)
        else:
            result = run_subprocess(argv, label=label, timeout_s=timeout_s, log_fh=log_fh)
        _hq_before = state.held_quat
        state.update_from_result(tool_name, result)
        snap_after = snapshot_robot_state()
        if state.held_quat is not None and state.held_quat != _hq_before:
            # the flange the chained orientation belongs to: the held pose in the flange is fixed by the grasp,
            # the world one is not (every later wrist turn - a hover yaw fix, a view - turns the part with it)
            state.held_tcp_quat = (snap_after.get('tcp') or {}).get('quat_xyzw')
        _record_snapshot(snap_fh, "after", label, snap_after)
        return result

    def say(msg):
        print(msg)
        if log_fh: log_fh.write(msg + "\n")

    executed = []           # (step_idx, step, tool_name, kw) in order, for the staging cycle
    placed_at_staging = False
    # Most recent recorded gripper width (mm). move_to_grasp applies it right before its
    # descent (--pre-grasp-width) instead of the gripper closing to it before the approach.
    grasp_width = {'mm': None}
    insert_recoveries = {'n': 0}
    seated_now = False                 # last step was a successful insert (the part is seated)

    def _release_width(idx):
        """Width of the recorded release right after step idx (the gripper command that follows)."""
        for st_ in sequence[idx:]:
            try:
                t_, k_ = parse_tool_call(st_)
            except Exception:
                continue
            if t_ == 'control_gripper' and _width(k_.get('command')) is not None:
                return _width(k_.get('command'))
        return grasp_width['mm']

    def _width(cmd):
        try:
            return float(cmd)
        except (TypeError, ValueError):
            return None

    def _prev_tool(idx):
        """Tool of the step before step idx (1-based), or None."""
        try:
            return parse_tool_call(sequence[idx - 2])[0] if idx >= 2 else None
        except Exception:
            return None

    grip_redone = {}

    def _next_tool(idx):
        try:
            return parse_tool_call(sequence[idx])[0] if idx < len(sequence) else None
        except Exception:
            return None

    skipping = args.skip_to is not None
    for step_idx, step in enumerate(sequence, start=1):
        try:
            tool_name, kw = parse_tool_call(step)
        except Exception as e:
            print(f"  [SKIP] step parse failed for {step!r}: {e}")
            continue

        # --skip-to <token>: skip steps until we hit a translate_object step
        # whose action matches the token (e.g. --skip-to insert), or any step
        # whose tool_name matches.
        if skipping:
            hit = False
            if tool_name == args.skip_to:
                hit = True
            elif tool_name == 'translate_object' and kw.get('action') == args.skip_to:
                hit = True
            if not hit:
                print(f"  [{step_idx}] SKIP (waiting for {args.skip_to!r}): {step}")
                continue
            skipping = False

        # --stop-before <token>: halt before the next matching step. Lets the
        # operator inspect robot state at a known pose for debugging.
        if args.stop_before is not None:
            stop_now = False
            if tool_name == args.stop_before:
                stop_now = True
            elif tool_name == 'translate_object' and kw.get('action') == args.stop_before:
                stop_now = True
            if stop_now:
                global _STOP_BEFORE_FIRED
                _STOP_BEFORE_FIRED = True
                msg = (f"\n  [{step_idx}] === STOP-BEFORE {args.stop_before!r} hit. "
                       f"Halting replay at {step!r}. Robot at last completed pose. ===\n")
                print(msg)
                if log_fh: log_fh.write(msg)
                return True  # treat as success — operator-requested halt

        # Recorded pre-grasp width step right before a grasp: defer it to move_to_grasp, which
        # sets it just before descending. Closing it here hid a part directly below the
        # gripper and made move_to_grasp open again to see (2026-10-07).
        if tool_name == 'control_gripper' and _width(kw.get('command')) is not None:
            grasp_width['mm'] = _width(kw.get('command'))
            if _next_tool(step_idx) == 'move_to_grasp':
                say(f"  [{step_idx}] pre-grasp width {grasp_width['mm']:g} mm deferred to move_to_grasp")
                continue

        if args.dry_run:
            try:
                argv, _ = build_argv(tool_name, kw, state, base_offset_xy=args.base_offset_xy,
                                     observed_base=getattr(args, 'observed_base', None))
            except Exception as e:
                print(f"  [{step_idx}] BUILD FAILED ({e}); aborting.")
                return False
            if tool_name == 'move_to_grasp' and base_name:
                argv += ['--base-name', base_name]
            if tool_name == 'move_to_grasp' and grasp_width['mm'] is not None:
                argv += ['--pre-grasp-width', f"{grasp_width['mm']:g}"]      # regrasp/recovery: as recorded
            print(f"  [{step_idx}] [DRY] {' '.join(argv)}")
            continue

        t_step = time.time()                       # insert logs written from here belong to this step
        if tool_name == 'translate_object' and kw.get('action') == 'insert' and not args.dry_run and not _SAFETY_STOP:
            pre_insert_base_read(obj, state, args, say, log_fh)  # fresh base read from the hover (base not fixed)
            held_yaw_calibration(obj, state, say, log_fh)      # one read where the arm is; no extra moves
        result = execute(step_idx, step, tool_name, kw)
        if result is None:
            return False
        # Grip-width guard: a close that ends far from the grasp's own width took the wrong thing
        # (2026-10-19 20:53: flat u_brown, fingers closed across both legs at 69.9 mm instead of
        # 21 on a leg; the part slipped out in the reorient and the release command failed).
        # verify_grasp only checks "closed on something". Open, lift, re-pick once.
        if (tool_name == 'control_gripper' and str(kw.get('command')) == 'close' and not _SAFETY_STOP
                and result.get('result') != 'failure' and result.get('final_width_mm') is not None
                and _prev_tool(step_idx) == 'move_to_grasp' and not grip_redone.get(step_idx)):
            try:
                from primitives.shared.config import grip_width_mm     # closed width on the part (u_brown 21, yellow 42)
                want = float(grip_width_mm(obj))
            except Exception:
                want = None
            got = float(result['final_width_mm'])
            if want is not None and abs(got - want) > GRIP_WIDTH_TOL_MM:
                grip_redone[step_idx] = True
                say(f"  [{step_idx}] grip {got:.1f} mm, expected {want:.1f} ± {GRIP_WIDTH_TOL_MM:g} -> wrong grasp; "
                    f"open, lift, re-pick once")
                redo = [('control_gripper', dict(command='open')),
                        ('move_to_safe_height', {}),
                        ('move_to_grasp', dict(object_name=obj, grasp_id=state.last_grasp_id or 1)),
                        ('control_gripper', dict(command='close'))]
                for j, (t_, k_) in enumerate(redo, start=1):
                    r_ = execute(step_idx, f"re-pick {t_}", t_, k_, tag=f"g{j}")
                    if r_ is None or r_.get('result') == 'failure':
                        result = r_ or {'result': 'failure', 'error': 're-pick build failed'}
                        break
                    time.sleep(0.5)
                else:
                    result = r_
                    got = float(result.get('final_width_mm') or 0)
                    if abs(got - want) > GRIP_WIDTH_TOL_MM:
                        say(f"  [{step_idx}] re-pick grip {got:.1f} mm, still not {want:.1f} -> stopping")
                        result = dict(result, result='failure', error='wrong grasp twice')

        # Staging regrasp: move_to_grasp has just turned top-down over the clear area
        # and observed the part before descending. If the face it landed on cannot be
        # inserted in one pass, run another place-down pass from here — the part is in
        # a fresh start state, exactly like the initial pick.
        if (tool_name == 'move_to_grasp' and placed_at_staging
                and result.get('result') != 'failure'):
            first = next((i for i, e in enumerate(executed) if e[2] == 'move_to_grasp'), None)
            cycle = executed[first + 1:] if first is not None else []
            for attempt in range(STAGING_MAX_RETRIES + 1):
                q = (result.get('current_object_orientation') or {}).get('quat')
                q = [q['x'], q['y'], q['z'], q['w']] if isinstance(q, dict) else None
                check = staging_single_pass_ok(obj, base_name, q)
                if check is None:
                    say(f"  [{step_idx}] staging check: could not evaluate {obj} landing — continuing")
                    break
                ok, err = check
                say(f"  [{step_idx}] staging check: {obj} up-face error {err:.1f} deg -> "
                    f"{'single pass OK' if ok else 'NEEDS ANOTHER PASS'}")
                if ok:
                    break
                if attempt == STAGING_MAX_RETRIES or not cycle:
                    say(f"\n!!! {obj} landing still not insertable after {attempt} extra pass(es)")
                    return False
                say(f"  [{step_idx}] {obj} toppled at staging — extra place-down pass "
                    f"{attempt + 1}/{STAGING_MAX_RETRIES}")
                for (i, st, t, k) in cycle + [(step_idx, step, tool_name, kw)]:
                    r = execute(i, st, t, k, tag=f"r{attempt + 1}")
                    if r is None or r.get('result') == 'failure':
                        print(f"\n!!! STAGING PASS STEP FAILED: {st}")
                        if r: print(json.dumps(r, indent=2))
                        return False
                    time.sleep(0.5)
                result = r   # the re-run regrasp's observation is checked next
            placed_at_staging = False

        executed.append((step_idx, step, tool_name, kw))
        if tool_name == 'translate_object' and kw.get('action') == 'place_down' \
                and result.get('result') != 'failure':
            placed_at_staging = True

        # --stop-after <token>: halt AFTER this step (and skip cleanup move_home).
        # Lets the operator finish a partial cycle (e.g. through insert) and then
        # take over with follow-on motion (e.g. move_to_safe_height keeping the
        # peg gripped, then iterate the insert in a regrasp loop).
        if args.stop_after is not None and result.get('result') != 'failure':
            stop_now = False
            if tool_name == args.stop_after:
                stop_now = True
            elif tool_name == 'translate_object' and kw.get('action') == args.stop_after:
                stop_now = True
            if stop_now:
                global _STOP_AFTER_FIRED
                _STOP_AFTER_FIRED = True
                msg = (f"\n  [{step_idx}] === STOP-AFTER {args.stop_after!r} hit. "
                       f"Halting replay after {step!r}. ===\n")
                print(msg)
                if log_fh: log_fh.write(msg)
                return True

        # A bar wedged just above its seat (2026-10-20 cycle 3: line_brown stalled 4.5-4.8 mm above the seat with
        # x/y within 0.2 mm of the aim, four times - a ~0.7 deg yaw in the jaws on a zero-clearance channel): do not
        # pull it out and repeat; release it where it is and let the seat check's nudges/taps finish it (how wedged
        # bars were seated before).
        def _wedge_check():
            """None: not wedged. True/False: wedged; released and nudged -> seated or not. Called after EVERY failed
            attempt, before any lift: lifting a held part out of a wedge tripped a protective stop (cycle 3)."""
            if not (_is_bar(obj) and not _SAFETY_STOP and not args.dry_run):
                return None
            w = _wedged_near_seat(obj, t_step)
            if w is None:
                return None
            say(f"  [{step_idx}] {obj} wedged {w['dz_mm']:.1f} mm above its seat, {w['dxy_mm']:.1f} mm from the aim "
                f"-> release here and let the seat check nudge it in")
            rel = _release_width(step_idx)
            ok_rel = all((execute(step_idx, f"wedged {t_}", t_, k_, tag=f"wedge{j}") or {}).get('result') != 'failure'
                         for j, (t_, k_) in enumerate([('control_gripper', dict(command=f"{rel:g}" if rel else 'open')),
                                                       ('move_to_safe_height', {})], start=1))
            if ok_rel and seat_check_and_reseat(obj, state, args, say):
                say(f"  [{step_idx}] {obj}: seated after the nudge")
                return True
            say(f"\n!!! {obj}: wedged and not seated by the nudges")
            return False

        if result.get('result') == 'failure' and tool_name == 'translate_object' and kw.get('action') == 'insert':
            wc = _wedge_check()
            if wc is not None:
                if wc:
                    print(f"\n  ✓ {obj} complete")
                return wc

        # Outer attempts (2026-10-20): the insert runs ONE attempt; between attempts the part is lifted, the base
        # re-read from a view the held part does not block, and the insert aims again. Blind in-insert retries
        # aimed at a base the previous attempt had pushed (fork_yellow slid base3 ~30 mm over cycle 2).
        k_try = 1
        while (result.get('result') == 'failure' and tool_name == 'translate_object' and kw.get('action') == 'insert'
               and k_try < INSERT_OUTER_TRIES and not _SAFETY_STOP and not args.dry_run):
            k_try += 1
            prev_ob = getattr(args, 'observed_base', None)
            say(f"  [{step_idx}] insert attempt {k_try - 1} failed -> lift, re-read the base, attempt {k_try}/{INSERT_OUTER_TRIES}")
            r_ = execute(step_idx, 'lift for a re-read', 'move_to_safe_height', dict(object_name=obj), tag=f"try{k_try}lift")
            if r_ is None or r_.get('result') == 'failure':
                break
            pre_insert_base_read(obj, state, args, say, log_fh)
            nb = getattr(args, 'observed_base', None)
            if prev_ob is not None and nb is not None:
                say(f"  [{step_idx}] base after attempt {k_try - 1}: moved "
                    f"{1000*np.hypot(nb[0][0]-prev_ob[0][0], nb[0][1]-prev_ob[0][1]):.1f} mm since the attempt's aim")
            result = execute(step_idx, step, tool_name, kw, tag=f"try{k_try}") or result
            if result.get('result') == 'failure':
                wc = _wedge_check()
                if wc is not None:
                    if wc:
                        print(f"\n  ✓ {obj} complete")
                    return wc

        if (result.get('result') == 'failure' and tool_name == 'translate_object'
                and kw.get('action') == 'insert' and insert_recoveries['n'] < INSERT_MAX_RECOVERIES
                and not _SAFETY_STOP):
            # A missed insert means a bad grasp (the insert itself searches ~5 mm). Set the
            # part down at the staging area and take a fresh grasp from close above, then
            # insert again — the operator's recovery that seated yellow on 2026-10-14.
            insert_recoveries['n'] += 1
            say(f"  [{step_idx}] insert failed -> recovery {insert_recoveries['n']}/{INSERT_MAX_RECOVERIES}: "
                f"place at staging, regrasp, reinsert")
            rel = _release_width(step_idx)
            rec = [('move_to_safe_height', {}),        # lift first: the clear-area move is joint-space
                   ('translate_object', dict(action='place_down', object_name=obj, base_name=base_name)),
                   ('control_gripper', dict(command=f"{rel:g}" if rel else 'open')),
                   ('move_to_safe_height', {}),
                   ('reread_base', {}),                # empty gripper: the failed attempts may have pushed the base
                   ('move_to_grasp', dict(object_name=obj, grasp_id=state.last_grasp_id or 1)),
                   ('control_gripper', dict(command='close')),
                   ('move_to_safe_height', dict(object_name=obj)),
                   ('rotate_object', dict(object_name=obj, base_name=base_name)),
                   ('translate_object', dict(kw))]
            for j, (t_, k_) in enumerate(rec, start=1):
                if t_ == 'reread_base':
                    # 2026-10-20 cycle 2: fork_yellow's three attempts slid base3 17.5 mm / 4.5 deg and the
                    # recovery's reinsert aimed at the read from before them. Re-read with the gripper empty.
                    nb = reread_base(args_base_name(args), getattr(args, 'observed_base', None), log_fh)
                    if nb is not None:
                        args.observed_base = nb
                    # back over the part we just set down - the re-read took the camera to the base, and the regrasp
                    # then waited for a part out of view (2026-10-20: line_brown sat at the clear area all along)
                    run_subprocess(['python3', '-u', '-m', 'primitives.core.move_to_clear_area', '--mode', 'real'],
                                   label='back over the set-down part', timeout_s=90, log_fh=log_fh)
                    _seen_here(obj, 10.0)                   # frames from after the arm stopped (still-frame rule)
                    continue
                r_ = execute(step_idx, f"recovery {t_}", t_, k_, tag=f"rec{j}")
                if t_ == 'translate_object' and r_ is not None and r_.get('result') == 'failure':
                    wc = _wedge_check()                    # never lift a held part out of a wedge
                    if wc is not None:
                        if wc:
                            print(f"\n  ✓ {obj} complete")
                        return wc
                if r_ is None or r_.get('result') == 'failure':
                    result = r_ or {'result': 'failure', 'error': 'recovery build failed'}
                    say(f"  [{step_idx}] recovery step {t_} failed")
                    break
                time.sleep(0.5)
            else:
                result = r_
                say(f"  [{step_idx}] recovery insert succeeded")

        if (result.get('result') == 'failure' and tool_name == 'control_gripper' and seated_now
                and not _SAFETY_STOP and not args.dry_run):
            # The release after an insert: the part is seated and still gripped. Retry it — and
            # if it will not open, never lift (the cleanup's safe-height move would pull the
            # seated part out; 2026-10-15 only a dropped robot program stopped it doing so).
            for k_ in range(2):
                time.sleep(2.0)
                say(f"  [{step_idx}] release failed ({result.get('error', '')[:80]}) -> retry {k_ + 1}/2")
                result = execute(step_idx, step, tool_name, kw, tag=f"retry{k_ + 1}") or result
                if result.get('result') != 'failure':
                    break
            if result.get('result') == 'failure':
                global _HOLDING_SEATED
                _HOLDING_SEATED = True
        if result.get('result') == 'failure':
            print(f"\n!!! STEP FAILED: {step}")
            print(json.dumps(result, indent=2))
            return False
        seated_now = False
        if tool_name == 'translate_object' and kw.get('action') == 'insert' and not args.dry_run:
            record_seat_offset(obj, t_step)
            seated_now = True

        # Tiny pause between primitives to let TF + topics settle (matches
        # run_assembly_step's settle-window discipline).
        time.sleep(0.5)

    if os.environ.get('SEAMS_CERTIFY') == '1' and not args.dry_run:
        # ExecuteMate is the authority (operator 2026-10-21: no diagnostics once the part is known seated). The
        # replay's own depth seat check and reseat run only as the recovery for a measured not_at_seat, or when no
        # view could measure the seat.
        bn_ = base_name or RUN_BASE['name']
        out = certify_via_execute_mate(obj, bn_, getattr(args, 'observed_base', None), say, log_fh) or {}
        if not out.get('placed') and _seat_checked(obj) and getattr(args, 'seat_check', True):
            say(f"  [execute_mate] {obj}: {out.get('refusal') or 'no answer'} -> the replay's seat check / reseat")
            if not seat_check_and_reseat(obj, state, args, say):
                return False
            if out.get('refusal') == 'not_at_seat':
                out = certify_via_execute_mate(obj, bn_, getattr(args, 'observed_base', None), say, log_fh) or {}
    elif _seat_checked(obj) and not args.dry_run and getattr(args, 'seat_check', True):
        if not seat_check_and_reseat(obj, state, args, say):
            return False
    print(f"\n  ✓ {obj} complete")
    return True


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument('--assembly-json', required=True,
                   help="Path to Assembly_<id>_results.json (sim-mode tool_sequences inside).")
    p.add_argument('--only', type=str, default=None,
                   help="Only replay this object's tool_sequence.")
    p.add_argument('--skip-to', type=str, default=None,
                   help="Skip steps until the first matching tool_name (or translate_object "
                        "action). Example: --skip-to insert  starts at the insert step.")
    p.add_argument('--stop-before', type=str, default=None,
                   help="Halt the replay BEFORE the first step whose tool_name (or "
                        "translate_object action) matches. Useful for debugging — stops "
                        "the robot at a known pose so the operator can introspect state. "
                        "Example: --stop-before rotate_object  runs through pick + grasp + "
                        "safe-height, then halts before the rotate_object step.")
    p.add_argument('--stop-after', type=str, default=None,
                   help="Halt the replay AFTER the first step whose tool_name (or "
                        "translate_object action) matches. Useful for finishing the insert "
                        "step then handing off to a follow-on motion. Example: "
                        "--stop-after insert  runs through pick + rotate + place + re-pick "
                        "+ rotate + insert, then halts (skipping the post-insert release + "
                        "safe-height retract). Skips the cleanup move_home as well.")
    p.add_argument('--seat-check', action=argparse.BooleanOptionalAction, default=True,
                   help="After line_green's insert, depth-check it is flat at seat height; if not, "
                        "take it with the insert grasp and reinsert (once). Default on.")
    p.add_argument('--base-from-camera', action=argparse.BooleanOptionalAction, default=True,
                   help="Aim inserts at the camera-observed base pose, measured once at startup "
                        "(default). --no-base-from-camera uses DEFAULT_BASE_POSITION + "
                        "per-object offsets as before.")
    p.add_argument('--base-offset-xy', type=float, nargs=2, metavar=('DX', 'DY'),
                   default=None,
                   help="xy offset (m) injected into the insert step's base pose. Useful for "
                        "tuning per-object insert location empirically.")
    p.add_argument('--derive-sequence', action=argparse.BooleanOptionalAction, default=True,
                   help="(default ON) Observe every part up front and run 1 or 2 passes as the "
                        "planner decides (compliant_insertion_studio/planning/"
                        "sequence_planner.py), instead of the recorded pass count. The camera "
                        "must see the parts: keep the startup (home + open) or position the arm. "
                        "--no-derive-sequence replays the recorded sequences as-is.")
    p.add_argument('--start-from', type=str, default=None, metavar='OBJECT',
                   help="Skip the parts before OBJECT in assembly order (they are already seated) "
                        "and run from OBJECT on, e.g. to resume after a stop.")
    p.add_argument('--post-insert-check', action='store_true',
                   help="After an object's sequence, read it from the post-release safe-height "
                        "pose and, if it sits off its correct-seat reference, retry minimally "
                        "(regrasp, lift 10 mm, insert). Objects in ablations/seat_check.py "
                        "SEAT_REF only (line_green). Off by default.")
    p.add_argument('--dry-run', action='store_true',
                   help="Print the planned subprocess argv without executing.")
    p.add_argument('--initial-held-quat', type=float, nargs=4,
                   metavar=('QX', 'QY', 'QZ', 'QW'), default=None,
                   help="Optional initial held_quat to seed the chain (e.g. when starting "
                        "mid-sequence with a part already in the gripper).")
    p.add_argument('--part-at-staging', action='store_true',
                   help='with --only/--skip-startup, arm over the clear area: the part waits in the '
                        'staging region (e.g. after disassemble.py --to-staging); after reading the base '
                        'the arm returns over the clear area instead of home')
    p.add_argument('--seat-check-only', default=None, metavar='PART',
                   help='PART is already seated: read the base, run the depth seat check and fix it '
                        '(nudge / reseat) as after an insert, then stop. No pick, no insert.')
    p.add_argument('--skip-startup', action='store_true',
                   help="Skip the move_home + gripper-open startup phase. Use when the "
                        "robot is already in a known clean state.")
    p.add_argument('--log-dir', type=str, default=str(LOG_ROOT),
                   help="Directory for the per-run log file. Default: ablations/logs/.")
    args = p.parse_args()
    if os.nice(0) < 5:                     # the UR driver has no RT priority here: run below it
        os.nice(5 - os.nice(0))

    # Install SIGINT handler so Ctrl-C kills the active subprocess tree
    # immediately instead of letting it run to completion (which previously
    # caused protective stops because the operator's interrupt didn't
    # propagate to the child primitive's motion).
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    if args.base_offset_xy is not None:
        args.base_offset_xy = tuple(args.base_offset_xy)

    if args.seat_check_only:
        # Arm empty, at home or safe height: re-read the base, then the same check-and-fix the
        # replay runs after an insert (the seat check flies to its own view).
        from disassemble import observe_base
        try:
            RUN_BASE['name'] = json.load(open(args.assembly_json)).get('base_name') or 'base1'
        except Exception:
            pass
        ob = observe_base(RUN_BASE['name'])
        if ob is None:
            print(f"{RUN_BASE['name']} not observed — stopping"); return 1
        args.observed_base = ob
        def say(m):
            print(m, flush=True)
        ok = seat_check_and_reseat(args.seat_check_only, AssemblyState(), args, say)
        run_subprocess(['python3', '-u', '-m', 'primitives.move_to_safe_height', '--mode', 'real'],
                       label='seat-check-only safe height', timeout_s=60)
        run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                       label='seat-check-only home', timeout_s=60)
        print(f"\n  seat-check-only {args.seat_check_only}: {'OK' if ok else 'NOT FIXED'}")
        return 0 if ok else 1

    json_path = Path(args.assembly_json)
    if not json_path.exists():
        print(f"ERROR: assembly JSON not found: {json_path}")
        return 2
    d = json.load(open(json_path))
    entries = sorted(d.get('assembly_order', []), key=lambda e: e.get('assembly_order', 0))
    if not entries:
        print("ERROR: no assembly_order entries in JSON")
        return 2
    RUN_BASE['name'] = d.get('base_name') or RUN_BASE['name']      # FMB3: base3 (every base1 default reads this)
    RUN_BASE['json'] = str(Path(args.assembly_json).resolve())     # child drivers (relocate) read the same assembly
    RUN_BASE['parts'] = tuple(e['object_name'] for e in entries)

    state = AssemblyState()
    if args.initial_held_quat is not None:
        state.held_quat = tuple(args.initial_held_quat)
        print(f"Initial held_quat seeded: {state.held_quat}")

    # Open per-run log file. Captures every primitive's stdout+stderr alongside
    # the orchestrator's headers. One persistent record per replay attempt so
    # we can post-mortem failures and force-stops without scraping terminal
    # scrollback.
    log_fh = None
    log_path = None
    snap_fh = None
    snap_path = None
    if not args.dry_run:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        only_tag = f"_{args.only}" if args.only else ""
        log_path = log_dir / f"replay{only_tag}_{ts}.log"
        snap_path = log_dir / f"replay{only_tag}_{ts}.snapshots.jsonl"
        log_fh = open(log_path, "w", buffering=1)  # line-buffered
        snap_fh = open(snap_path, "w", buffering=1)
        _start_timing(log_dir / f"replay{only_tag}_{ts}.timing.jsonl", log_dir / f"replay{only_tag}_{ts}.motion.jsonl")
        log_fh.write(
            f"# replay_real_assembly log\n"
            f"# started: {_dt.datetime.now().isoformat()}\n"
            f"# argv: {sys.argv}\n"
            f"# assembly_json: {args.assembly_json}\n"
            f"# only: {args.only}\n"
            f"# base_offset_xy: {args.base_offset_xy}\n"
            f"# skip_startup: {args.skip_startup}\n"
            f"# stop_before: {args.stop_before}\n"
            f"# initial_held_quat: {args.initial_held_quat}\n"
            f"# snapshots: {snap_path}\n"
            f"# {'='*70}\n"
        )
        print(f"Logging to: {log_path}")
        print(f"Joint snapshots: {snap_path}")

    try:
        # Startup: clean state — move home + open gripper. Skipped via --skip-startup.
        if not args.skip_startup and not args.dry_run:
            header = f"\n{'='*72}\n  STARTUP: move_home + gripper-open\n{'='*72}\n"
            print(header, end='')
            if log_fh: log_fh.write(header)
            startup_steps = [
                (['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                 'startup move_home', 60),
                (['python3', '-u', '-m', 'primitives.control_gripper', '100', '--mode', 'real'],
                 'startup gripper-open', 30),
            ]
            for argv, label, timeout_s in startup_steps:
                r = run_subprocess(argv, label=label, timeout_s=timeout_s, log_fh=log_fh)
                if r.get('result') == 'failure':
                    fail_msg = f"\n!!! STARTUP FAILED at {label}\n{json.dumps(r, indent=2)}\n"
                    print(fail_msg)
                    if log_fh: log_fh.write(fail_msg)
                    return 1

        # --derive-sequence: plan EVERY part up front, from wherever the startup left the
        # arm (home, gripper open — all table parts in view) or the operator put it. Start
        # poses do not change until a part is picked, and after an insert the arm sits over
        # the base where the next part may be out of view (a 30 s observe timeout each).
        # Where the disassembly set parts down (disassemble.record_staged, < 3 h old): the last-seen position
        # for a part no view sees, so _part_view looks straight above it.
        try:
            _st = json.load(open(REPO_ROOT / 'ablations/logs/staged_parts.json'))
            for _n, _v in _st.items():
                _x, _y, _t = _v[:3]                     # entries may carry the placed pose after (x, y, t)
                if time.time() - _t < 3 * 3600:
                    LAST_SEEN_XY.setdefault(_n, (_x, _y))
        except Exception:
            pass
        if getattr(args, 'derive_sequence', False) and not args.dry_run:
            names = [e.get('object_name') for e in entries]
            skip = set(names[:names.index(args.start_from)]) if getattr(args, 'start_from', None) in names else set()
            unseen_home = []
            for entry in entries:
                obj = entry.get('object_name')
                if (args.only and obj != args.only) or obj in skip:
                    continue
                seq = entry.get('tool_sequence', []) or []
                bn = None
                for s_ in seq:
                    try:
                        bn = parse_tool_call(s_)[1].get('base_name') or bn
                    except Exception:
                        pass
                entry['tool_sequence'], note = derive_sequence_for(obj, seq, bn, timeout_s=10.0)
                if note.startswith('observe failed'):
                    unseen_home.append((entry, seq, bn))          # not where we usually keep it
                    continue
                msg = f"  [derive-sequence] {obj}: {note}"
                print(msg)
                if log_fh: log_fh.write(msg + "\n")
            # Parts home cannot see: the generated-layout disassembly puts them in the staging region
            # (staging_layout.py; cell_layout.yaml). Plan those from the staging view, then back home -
            # or from above the clear area when the part waits there (--part-at-staging; 2026-10-20 the
            # planner looked at FMB1's parked set for an FMB3 part sitting at the clear area).
            def _unseen_view():
                if args.part_at_staging:
                    r_ = run_subprocess(['python3', '-u', '-m', 'primitives.core.move_to_clear_area', '--mode', 'real'],
                                        label='view the clear area', timeout_s=90, log_fh=log_fh)
                    time.sleep(1.5)
                    return r_.get('result') != 'failure'
                return _staging_view(log_fh)
            if unseen_home and _unseen_view():
                for entry, seq, bn in unseen_home:
                    entry['tool_sequence'], note = derive_sequence_for(entry['object_name'], seq, bn, timeout_s=15.0)
                    msg = f"  [derive-sequence] {entry['object_name']} (staging view): {note}"
                    print(msg)
                    if log_fh: log_fh.write(msg + "\n")
                run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                               label='back home after the staging view', timeout_s=60, log_fh=log_fh)

        # Base pose for the inserts: read here (startup view), re-read between parts (reread_base).
        args.observed_base = None
        bname = 'base1'
        if getattr(args, 'base_from_camera', False) and not args.dry_run:
            for e_ in entries:
                for s_ in e_.get('tool_sequence', []) or []:
                    try:
                        bname = parse_tool_call(s_)[1].get('base_name') or bname
                    except Exception:
                        pass
            args.base_name = bname                 # every later base1 default reads this (FMB3: base3)
            RUN_BASE['name'] = bname
            # Read the base from straight above, then back home so the picks start with every
            # table part in view. Skipped (home-view read) if the move is refused.
            # x, y and yaw come from the DEPTH outline of the base's top face (base_depth.py):
            # the marker read moved 55-61 mm in x between views and runs while the parts seated
            # at ~61, and line_green, aimed 6.4 mm short, could not reach its slot (2026-10-15).
            # Flange 0.56 m: the D415 returns no depth closer than ~0.45 m.
            from primitives.shared.config import DEFAULT_BASE_POSITION as _DBP
            from base_depth import VIEW_FLANGE_Z as _VZ, measure as _base_from_depth
            # over the assembly region of the cell layout (primitives/shared/cell_layout.yaml), wherever
            # the run's base is: DEFAULT_BASE_POSITION is base1's calibration spot
            from primitives.shared.config import ASSEMBLY_REGION_CENTER as _ARC, CELL_LAYOUT_NAME as _LN
            _lb = _last_base_pose(bname)          # over the base where it was last read (it moves: 9 cm off the
            _bx, _by = (_lb[0][0], _lb[0][1]) if _lb is not None else _ARC     # region centre left it half out of view)
            msg = f"  [layout] {_LN}: base view over the assembly region ({_bx:.3f}, {_by:.3f})"
            print(msg, flush=True)
            if log_fh: log_fh.write(msg + "\n")
            r_ = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                 f"{_bx:.4f}", f"{_by:.4f}", f"{_VZ:.2f}"],
                                label='base-view move', timeout_s=60, log_fh=log_fh)
            above = r_.get('result') == 'success'
            ob = observe_base_pose(bname)
            if above and ob is not None:
                time.sleep(1.0)                                # settle before the depth frames
                ob_d, info_d = _base_from_depth(ob, bname)
                msg = f"  [base] depth outline: {json.dumps(info_d)}"
                print(msg)
                if log_fh: log_fh.write(msg + "\n")
                # 2026-10-20: after the hand-eye refit the markers are the calibrated read (1.2 mm across 24
                # views); the depth outline sits ~5 mm off in x and the validation insert seated -3.7 mm in x
                # from it. Markers by default; BASE_FROM=depth restores the outline.
                ob = choose_base_read(bname, ob, ob_d, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")))
            if above:
                # --part-at-staging: the part waits in the clear area, which home cannot see —
                # go back over it (top-down, orientation kept) instead of home.
                back = (['primitives.core.move_to_clear_area', '--mode', 'real'] if args.part_at_staging
                        else ['primitives.move_home', '--mode', 'real'])
                r_ = run_subprocess(['python3', '-u', '-m', *back],
                                    label='base-view return ' + ('to staging' if args.part_at_staging else 'home'),
                                    timeout_s=90, log_fh=log_fh)
                if r_.get('result') == 'failure':
                    print("\n!!! base-view return FAILED — stopping"); return 1
            if ob is None:
                msg = f"  [base] {bname} not observed -> DEFAULT_BASE_POSITION + per-object offsets"
            else:
                args.observed_base = ob
                _save_base_pose(bname, ob)
                msg = (f"  [base] {bname} observed at ({1000*ob[0][0]:.1f}, {1000*ob[0][1]:.1f}, "
                       f"{1000*ob[0][2]:.1f}) mm, yaw {np.degrees(2*np.arctan2(ob[1][2], ob[1][3])):.2f} deg "
                       f"-> inserts aim at it (per-object z offsets kept)")
            print(msg)
            if log_fh: log_fh.write(msg + "\n")
            # Spacing pre-check before any pick: every table part must stand clear of base1 and of
            # the other parts by an open finger's reach (spacing_check.py). Home sees the table;
            # with --part-at-staging the part waits in the clear area and home is not where we are.
            if ob is not None and not args.part_at_staging:
                from spacing_check import check as _spacing, observe_parts as _obs_parts, PARTS as _PARTS
                _PARTS = RUN_BASE.get('parts') or _PARTS           # this assembly's parts, around its base
                seen = _obs_parts(_PARTS)
                ok_sp, rep = _spacing(seen, args.observed_base or ob, base=RUN_BASE['name'])
                rep['unseen'] = [n for n in _PARTS if n not in seen]
                msg = f"  [spacing] {json.dumps(rep)}"
                print(msg, flush=True)
                if log_fh: log_fh.write(msg + "\n")
                if not ok_sp:
                    # A part too close is moved, not a reason to stop (operator 2026-10-19): the
                    # disassembly's --relocate (clearest grasp at its own opening; a spot in the staging
                    # region when the part is there, else a free spot seen from home), then re-check once.
                    pair = (rep.get('too_close') or [[None, None]])[0]
                    cands = [x for x in pair[:2] if x and not x.startswith('base') and x in seen]
                    from primitives.shared.config import STAGING_REGION as _SR
                    in_staging = [x for x in cands if _SR['x'][0] <= seen[x][0][0] <= _SR['x'][1]
                                  and _SR['y'][0] <= seen[x][0][1] <= _SR['y'][1]]
                    mover = (in_staging or cands or [None])[0]
                    if mover is None:
                        print("\n!!! parts too close for the open gripper and none of the pair can be moved — stopping")
                        return 1
                    msg = f"  [spacing] {pair[0]} / {pair[1]} too close -> relocating {mover}"
                    print(msg, flush=True)
                    if log_fh: log_fh.write(msg + "\n")
                    r_ = run_subprocess([sys.executable, '-u', str(REPO_ROOT / 'ablations/disassemble.py'), '--assembly-json', RUN_BASE.get('json', 'x'),
                                         '--relocate', mover, '--grasp-id', 'auto'] + (['--spot-staging'] if in_staging else []),
                                        label=f'relocate {mover}', timeout_s=900, log_fh=log_fh)
                    if r_.get('result') == 'failure':
                        print(f"\n!!! relocating {mover} failed — stopping"); return 1
                    seen = _obs_parts(_PARTS)
                    ok_sp, rep = _spacing(seen, args.observed_base or ob)
                    rep['unseen'] = [n for n in _PARTS if n not in seen]
                    msg = f"  [spacing] after relocating {mover}: {json.dumps(rep)}"
                    print(msg, flush=True)
                    if log_fh: log_fh.write(msg + "\n")
                    if not ok_sp:
                        print("\n!!! parts still too close after one relocation — stopping"); return 1

        completed: list[str] = []
        failed: Optional[str] = None
        first_object = True
        started = getattr(args, 'start_from', None) is None
        for entry in entries:
            obj = entry.get('object_name')
            if args.only and obj != args.only:
                continue
            # --start-from: resume mid-assembly (earlier parts already seated).
            if not started:
                if obj != args.start_from:
                    continue
                started = True
            # Between parts: go home first. Each sequence ends with move_to_safe_height
            # over the base, and from there the camera cannot see the parts still on the
            # table — 2026-10-07 the one-command run failed u_orange's pick with "not
            # detected (out of view)". Safe height already happened (end of the previous
            # sequence), so a straight move_home here is within the CLAUDE.md rule.
            if not first_object and not args.dry_run:
                hdr = f"\n  BETWEEN PARTS: re-read the base, then move_home before {obj} (camera view of the table)\n"
                print(hdr, end='')
                if log_fh: log_fh.write(hdr)
                if getattr(args, 'observed_base', None) is not None:
                    args.observed_base = reread_base(bname, args.observed_base, log_fh=log_fh)
                    # Arm is above the base: confirm every part seated so far really is.
                    seated_so_far = [e_.get('object_name') for e_ in entries if e_.get('object_name') in completed]
                    if seated_so_far and not verify_seated(seated_so_far, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")),
                                                                second_view=lambda: _home_view(log_fh)):
                        failed = seated_so_far[-1]
                        break
                r = run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                                   label=f'between-parts move_home ({obj})', timeout_s=60,
                                   log_fh=log_fh)
                if r.get('result') == 'failure':
                    failed = obj
                    break
            first_object = False
            if not args.dry_run and not _SAFETY_STOP and obj in FOLLOWS_SEATED:
                record_follow_reads(obj, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")),
                                    getattr(args, 'observed_base', None), log_fh, bool(args.part_at_staging))
            # --part-at-staging is a claim, not a fact (2026-10-20: line_brown was 38 cm away, in plain view from
            # home, and move_to_grasp gave up after 40 tries at the clear area): not seen there -> home, then above
            # its last position / the staging view.
            if (not args.dry_run and args.part_at_staging and not _SAFETY_STOP and not _seen_here(obj)):
                msg = f"  [view] {obj}: not at the clear area -> look from home, then the staging view"
                print(msg, flush=True)
                if log_fh: log_fh.write(msg + "\n")
                run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                               label='look from home', timeout_s=60, log_fh=log_fh)
                time.sleep(1.0)
                if not _seen_here(obj, 10.0) and not _part_view(obj, log_fh):
                    msg = f"\n!!! {obj}: not at the clear area, not seen from home nor the staging view"
                    print(msg, flush=True)
                    if log_fh: log_fh.write(msg + "\n")
                    failed = obj
                    break
            # Not where we usually keep it (home view)? Look over the staging region and pick from
            # there (operator 2026-10-19: move over the staging region as part of the assembly).
            if (not args.dry_run and not args.part_at_staging and not _SAFETY_STOP
                    and not _seen_here(obj)):
                msg = f"  [view] {obj}: not seen from home -> look above it"
                print(msg, flush=True)
                if log_fh: log_fh.write(msg + "\n")
                if not _part_view(obj, log_fh):
                    msg = f"\n!!! {obj}: not seen from above its last position nor from the staging view"
                    print(msg, flush=True)
                    if log_fh: log_fh.write(msg + "\n")
                    failed = obj
                    break
            # Room for the recorded grasp's open fingers where the part lies (relocates it if not)
            if not args.dry_run and not args.part_at_staging and not _SAFETY_STOP:
                gid_ = next((parse_tool_call(s_)[1].get('grasp_id') for s_ in entry.get('tool_sequence', []) or []
                             if 'move_to_grasp' in s_), None)
                # the fingers come down at the recorded PRE-GRASP opening (fmb1_assembly.json), not the grip width
                from primitives.shared.config import get_gripper_width_mm as _ggw   # any assembly (was FMB1's file)
                open_mm = _ggw(obj, int(gid_)) if gid_ is not None else None
                if gid_ is not None and open_mm and not _pick_room(obj, int(gid_), float(open_mm), log_fh,
                                                                     base_pose=getattr(args, 'observed_base', None)):
                    failed = obj
                    break
                if obj not in LAST_SEEN_XY and not _seen_here(obj) and not _part_view(obj, log_fh):
                    failed = obj
                    break
            ok = replay_object(entry, state, args, log_fh=log_fh, snap_fh=snap_fh)
            # Post-insert seat check (opt-in): the sequence ends at the post-release
            # safe-height pose, which is where the correct-seat references were recorded.
            if ok and getattr(args, 'post_insert_check', False) and not args.dry_run:
                import seat_check
                if obj in seat_check.SEAT_REF:
                    from primitives.shared.config import get_gripper_width_mm
                    w = get_gripper_width_mm(obj, seat_check.SEAT_REF[obj]['grasp_id'])
                    def _say(m):
                        print(m)
                        if log_fh: log_fh.write(m + "\n")
                    ok = seat_check.run(obj, w, say=_say)
            if ok:
                completed.append(obj)
            else:
                failed = obj
                break

        # Cleanup: move_home (always, success or failure path). Skipped on
        # --dry-run AND when --stop-before / --stop-after fired (pause means
        # the robot stays at the halted pose for operator inspection or
        # follow-on motion).
        # Final check of the whole assembly from the base view (the between-parts checks covered
        # the earlier parts; this one adds the last part and re-checks the rest).
        if (not failed and completed and not args.dry_run and not _SAFETY_STOP
                and getattr(args, 'observed_base', None) is not None):
            args.observed_base = reread_base(bname, args.observed_base, log_fh=log_fh)
            seated_all = [e_.get('object_name') for e_ in entries if e_.get('object_name') in completed]
            if not verify_seated(seated_all, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")),
                                 second_view=lambda: _home_view(log_fh)):
                failed = seated_all[-1]
                completed = [c for c in completed if c != failed]

        if _HOLDING_SEATED:
            msg = ("\n  CLEANUP SKIPPED: the gripper is closed on a SEATED part and would not open — no lift "
                   "(it would pull the part out). Open it by hand or: python3 -u -m primitives.control_gripper "
                   "open --mode real; then move_to_safe_height.\n")
            print(msg, end='')
            if log_fh: log_fh.write(msg)
        elif _SAFETY_STOP:
            msg = (f"\n  CLEANUP SKIPPED: robot is in {_SAFETY_STOP}. Clear it (dashboard 'close safety popup',"
                   f" 'unlock protective stop'), run stop_run.sh, then lift with move_to_safe_height.\n")
            print(msg, end='')
            if log_fh: log_fh.write(msg)
        elif not args.dry_run and not _STOP_BEFORE_FIRED and not _STOP_AFTER_FIRED:
            header = f"\n{'='*72}\n  CLEANUP: move_to_safe_height -> move_home\n{'='*72}\n"
            print(header, end='')
            if log_fh: log_fh.write(header)
            # Safe height FIRST (CLAUDE.md hard constraint). A failed insert can leave the
            # gripper closed on a part in the base — translate_object passes
            # --no-post-insert-move, so the wrapper's own retract is skipped — and
            # move_home plans a straight line that ignores seated parts. 2026-10-07: it
            # pulled a seated u_orange out of base1 and carried it home.
            run_subprocess(
                ['python3', '-u', '-m', 'primitives.move_to_safe_height', '--mode', 'real'],
                label='cleanup move_to_safe_height', timeout_s=60, log_fh=log_fh,
            )
            run_subprocess(
                ['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                label='cleanup move_home', timeout_s=60, log_fh=log_fh,
            )
        elif _STOP_BEFORE_FIRED or _STOP_AFTER_FIRED:
            which = 'stop-before' if _STOP_BEFORE_FIRED else 'stop-after'
            msg = f"\n  CLEANUP: move_home SKIPPED (--{which} fired; leaving robot at halted pose).\n"
            print(msg, end='')
            if log_fh: log_fh.write(msg)

        summary = f"\n{'='*72}\n  REPLAY SUMMARY\n{'='*72}\n  Completed: {completed}\n"
        if failed:
            summary += f"  FAILED at: {failed}\n"
        else:
            summary += f"  All done.\n"
        if log_path:
            summary += f"  Log: {log_path}\n"
        print(summary, end='')
        if log_fh: log_fh.write(summary)
        return 1 if failed else 0
    finally:
        for fh in (log_fh, snap_fh):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Timing (2026-10-21, Speed track T0): one JSONL row per subprocess / read / seam call / part, plus an independent
# motion recorder (TCP speed). ablations/cycle_time.py splits the wall time into arm-moving and arm-still.
# Logging only: nothing here changes what the replay does.
# ---------------------------------------------------------------------------
_TIMING = {'fh': None, 'rec': None}


def _start_timing(timing_path, motion_path):
    import atexit
    _TIMING['fh'] = open(timing_path, 'a', buffering=1)
    try:
        _TIMING['rec'] = subprocess.Popen(['python3', '-u', str(REPO_ROOT / 'ablations' / 'motion_recorder.py'),
                                           str(motion_path)], env=make_env(), stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)
    except Exception:
        pass
    try:
        _tn = 'part_' + 'tracker.py'                     # built at runtime: pgrep must not match this process
        if subprocess.run(['pgrep', '-f', _tn], capture_output=True).returncode == 0:
            raise RuntimeError('a tracker is already running (standing process)')
        _TIMING['tracker'] = subprocess.Popen(['python3', '-u', str(REPO_ROOT / 'ablations' / 'part_tracker.py')],
                                              env=make_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    print(f"  [timing] {timing_path}", flush=True)
    atexit.register(_stop_timing)


def _stop_timing():
    for k in ('rec', 'tracker'):
        r = _TIMING.get(k)
        if r is not None and r.poll() is None:
            r.send_signal(signal.SIGINT)
            try:
                r.wait(timeout=5)
            except Exception:
                r.terminate()


def _timed(fn, kind, label_of):
    import functools

    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.time()
        try:
            out = fn(*a, **k)
            return out
        finally:
            fh = _TIMING.get('fh')
            if fh is not None:
                try:
                    lab = label_of(a, k)
                except Exception:
                    lab = '?'
                row = {'kind': kind, 'label': lab, 't0': t0, 't1': time.time()}
                o = locals().get('out')
                if isinstance(o, dict) and o.get('duration_s') is not None:
                    row['commanded_s'] = o.get('duration_s')
                if isinstance(o, dict) and 'result' in o:
                    row['result'] = o.get('result')
                fh.write(json.dumps(row, default=str) + '\n')
    return w


run_subprocess = _timed(run_subprocess, 'subprocess', lambda a, k: k.get('label') or (a[1] if len(a) > 1 else '?'))
observe_object_orientation = _timed(observe_object_orientation, 'observe', lambda a, k: a[0])
observe_base_pose = _timed(observe_base_pose, 'observe_base', lambda a, k: a[0])
camera_brightness = _timed(camera_brightness, 'brightness', lambda a, k: '')
_seam_call = _timed(_seam_call, 'seam', lambda a, k: ' '.join(a[0][:3]))
replay_object = _timed(replay_object, 'part', lambda a, k: a[0]['object_name'])
seat_check_and_reseat = _timed(seat_check_and_reseat, 'seat_check', lambda a, k: a[0])
observe_many = _timed(observe_many, 'observe_many', lambda a, k: ','.join(a[0]) if a else '')
held_yaw_calibration = _timed(held_yaw_calibration, 'held_yaw', lambda a, k: a[0])
pre_insert_base_read = _timed(pre_insert_base_read, 'pre_insert_read', lambda a, k: a[0])
verify_seated = _timed(verify_seated, 'verify', lambda a, k: ','.join(a[0]) if a else '')


if __name__ == '__main__':
    sys.exit(main())
