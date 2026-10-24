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


STATE_TAP_JSON = REPO_ROOT / 'ablations' / 'logs' / 'robot_state_latest.json'
STATE_TAP_FRESH_S = 0.3


def snapshot_robot_state(timeout_s: float = 2.0) -> dict:
    """Grab a single sample of /joint_states + /tcp_pose_broadcaster/pose. From the standing state tap
    (ablations/state_tap.py, rewritten at 20 Hz) when its file is fresh; otherwise by spawning a tiny rclpy
    subscriber subprocess (~0.8-1 s, twice per step - the median 2.24 s between steps, 2026-10-21). Returns
    {joints: ..., tcp: ...} or {error: ...} on failure. Out-of-process so we don't conflict with any
    long-running rclpy contexts in the orchestrator."""
    try:
        d = json.loads(STATE_TAP_JSON.read_text())
        if time.time() - float(d.get('t', 0)) <= STATE_TAP_FRESH_S and d.get('joints') and d.get('tcp'):
            return {'joints': d['joints'], 'tcp': d['tcp']}
    except Exception:
        pass
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
SATURATED_BRIGHTNESS = 230.0     # above: the frame is blown out (10-22 01:55: room lights on with the dark-room manual
                                 # exposure still set - brightness 254, no marker anywhere, the yellow 'not found')
DARK_EXPOSURE = ('8000', '128')  # manual exposure, gain that restore marker reads with the lights off (dim-light memo)


EXPOSURE_RECHECK_S = 60.0
_EXPOSURE_LAST = {}


def ensure_exposure(say=print):
    """Keep the D415 colour exposure usable: blown out -> auto exposure; very dark under auto -> the dark-room manual
    exposure. Returns the brightness after any change (None if no frame). A normal reading under EXPOSURE_RECHECK_S old is
    reused: the room's light does not swing between the extremes in a minute, and each read is a camera process (~1 s;
    run 44 spent 9.2 s still on them)."""
    c = _EXPOSURE_LAST
    if c.get('b') is not None and time.time() - c['t'] < EXPOSURE_RECHECK_S:
        return c['b']
    b = camera_brightness()
    if b is not None and 12 <= b <= SATURATED_BRIGHTNESS:
        c.update(b=b, t=time.time())
    if b is None or b < 0:
        return b
    def _set(*kv):
        for k, v in kv:
            subprocess.run(['ros2', 'param', 'set', '/camera/camera', k, v], capture_output=True, timeout=10, env=make_env())
    if b > SATURATED_BRIGHTNESS:
        _set(('rgb_camera.enable_auto_exposure', 'true')); time.sleep(1.5)
        b2 = camera_brightness(); say(f"  [camera] frame blown out ({b:.0f}) -> auto exposure ({b2})"); return b2
    if b < 12:
        _set(('rgb_camera.enable_auto_exposure', 'false'), ('rgb_camera.exposure', DARK_EXPOSURE[0]),
             ('rgb_camera.gain', DARK_EXPOSURE[1])); time.sleep(1.5)
        b2 = camera_brightness(); say(f"  [camera] very dark ({b:.0f}) -> manual exposure {DARK_EXPOSURE[0]} / gain {DARK_EXPOSURE[1]} ({b2})"); return b2
    return b


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



# Parts a later part hides from the camera once seated (operator, 2026-10-17: after yellow, line_green
# is not visible — accept it from the ledger if it was verified before yellow covered it).
OCCLUDED_BY = {'line_green': ('inverted_u_yellow',)}
VERIFY_READS = 3                 # verify_assembly readings per part; majority (of detections) decides
VERIFY_MAX_READS = 8             # "not detected" reads extend the window up to this many


from seams.seated_parts import SeatedParts   # noqa: E402  (AR4: the one writer of the ledger + seat offsets)
SEATED = SeatedParts()


def _ledger():
    return SEATED.ledger()


def forget_verified(obj: str) -> None:
    SEATED.forget(obj, offset=False)


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
PLACED_T = {}                     # part -> time it was last set down by the arm: older tracker sightings are stale
PART_VIEW_Z = 0.52                # flange height for a look straight above one part (RGB only; depth not needed)
PART_VIEW_Z_LOW_LIGHT = 0.40      # in low light the markers need the closer view (2026-10-20)


SEEN_SAMPLES: dict = {}           # obj -> (time, samples) of the last _seen_here that saw it


def _seen_here(obj, timeout_s=5.0) -> bool:
    ensure_exposure(lambda m: print(m, flush=True))
    obs = observe_object_orientation(obj, timeout_s, n_samples=2)
    ok = bool(obs.get('samples')) and not obs.get('error')
    if ok:
        SEEN_SAMPLES[obj] = (time.time(), obs['samples'])
    if not ok:
        # dark room (10-22 01:49: brightness ~31): markers need the 0.40 m view - lower straight down (top-down camera,
        # flange still 0.40 m over the table) and look again before declaring the part absent
        b_ = camera_brightness(); cur = (snapshot_robot_state().get('tcp') or {})
        xyz_, q_ = cur.get('xyz'), cur.get('quat_xyzw')
        if (b_ is not None and 0 <= b_ < LOW_LIGHT_BRIGHTNESS and xyz_ and q_ and xyz_[2] > PART_VIEW_Z_LOW_LIGHT + 0.02
                and abs(abs(q_[1]) - 1.0) < 0.02):
            r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                f"{xyz_[0]:.4f}", f"{xyz_[1]:.4f}", f"{PART_VIEW_Z_LOW_LIGHT:.4f}", '--quat', '0.0', '1.0',
                                '0.0', '0.0', '--min-z', '0.15'], label=f'low-light look for {obj} (0.40 m)', timeout_s=60)
            if r.get('result') != 'failure':
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


PLACED_JSON = REPO_ROOT / 'ablations' / 'logs' / 'placed_parts.json'   # part -> where the ARM last set it down (any process)
CLEAR_OCCUPIED_M = 0.09          # another part this close to the clear area blocks a set-down there (parts are <= 70 mm)


def placed(obj):
    """(x, y, t) where the arm last set `obj` down (written by place_down in any replay process), or None."""
    try:
        e = json.load(open(PLACED_JSON)).get(obj)
        return (float(e['xy'][0]), float(e['xy'][1]), float(e['t']))
    except Exception:
        return None


def record_placed(obj, xy):
    try:
        d = json.load(open(PLACED_JSON)) if PLACED_JSON.exists() else {}
    except Exception:
        d = {}
    d[obj] = {'xy': [round(float(xy[0]), 4), round(float(xy[1]), 4)], 't': time.time()}
    tmp = PLACED_JSON.with_suffix('.tmp'); json.dump(d, open(tmp, 'w'), indent=1); os.replace(tmp, PLACED_JSON)


def tracked(obj, max_age_s=TRACKER_MAX_AGE_S):
    """(x, y, age_s) of `obj`'s latest still-camera sighting from ANY view (ablations/part_tracker.py), or None.
    A set-down by the arm newer than the sighting wins (10-22: the regrasp looked at the part's old pick spot twice -
    in the run that placed it, and again in the next process)."""
    pl = placed(obj)
    try:
        e = json.load(open(TRACKER_JSON)).get(obj)
        t_seen = float(e['t'])
    except Exception:
        e, t_seen = None, 0.0
    if pl is not None and pl[2] >= t_seen:
        return (pl[0], pl[1], time.time() - pl[2])
    if e is None:
        return None
    age = time.time() - t_seen
    return (float(e['xyz'][0]), float(e['xyz'][1]), age) if age <= max_age_s else None


PART_AWAY_FROM_CLEAR_M = 0.08   # a last sighting this far from the clear area: the part is not there


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
    b_ = ensure_exposure(lambda m: print(m, flush=True))
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
            e_ = r.get('position_error_m')
            if isinstance(e_, dict) and 'x' in e_ and 'y' in e_:   # base-frame xy error: the aim's follow reads it
                SEATED.set_seat_error(o, [1000 * float(e_['x']), 1000 * float(e_['y'])], 'marker base view')
            led[o] = SEATED.mark(o, True, how='marker xy %.1f mm' % (1000 * r.get('position_error_judged_m', 0)))
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
        led[o] = SEATED.mark(o, False)
        ok_all = False
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
        from primitives.shared.config import held_tool_offset, held_tool_offset_for
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
    depth_done = set()
    if observed_base is not None:
        # CAD + depth first (operator 2026-10-21): the gap between each part's legs, measured from one still depth view
        # of the base against the CAD gap. The marker close read called u_brown +0.6 mm off; depth put its gap +3.0 mm
        # (-10..+16 vs CAD -13..+13) and line_green, aimed at CAD, landed on the leg every attempt.
        try:
            (bx_, by_, bz_), _ = observed_base
            r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                f"{bx_:.4f}", f"{by_:.4f}", "0.56", '--quat', '0.0', '1.0', '0.0', '0.0'],
                               label='follow read: depth view of the base', timeout_s=60, log_fh=log_fh)
            if r.get('result') == 'success':
                time.sleep(1.0)
                from notch_depth import measure as _notches
                nd = _notches(RUN_BASE['name'], observed_base, list(parts))
                for p, v in (nd.items() if isinstance(nd, dict) else []):
                    if isinstance(v, dict) and 'dx_m' in v:
                        SEATED.set_seat_error(p, [1000 * float(v['dx_m']), 0.0], 'notch depth')
                        led.setdefault(p, {})['rel_err_mm'] = [1000 * float(v['dx_m']), 0.0]
                        depth_done.add(p)
                        say(f"  [follow] {p}: gap {v['gap_mm']} mm vs CAD {v['cad_gap_mm']} (width {v['width_mm']}) -> "
                            f"{1000*v['dx_m']:+.1f} mm across (depth vs CAD)")
                    else:
                        say(f"  [follow] {p}: depth gap not measured ({(v or {}).get('error') if isinstance(v, dict) else v})")
        except Exception as e:
            say(f"  [follow] depth gap read failed ({e}) - marker close reads")
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        (bx, by, bz), bq = observed_base
        Rb = R_.from_quat(bq)
        for p in [p_ for p_ in parts if p_ not in depth_done]:
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
            SEATED.set_seat_error(p, [1000 * float(e[0]), 1000 * float(e[1])], 'marker close read')
            led.setdefault(p, {})['rel_err_mm'] = [1000 * float(e[0]), 1000 * float(e[1])]
            say(f"  [follow] {p}: seen ({1000*e[0]:+.1f}, {1000*e[1]:+.1f}) mm from its CAD seat (base frame, close read)")
    for p in parts:
        if isinstance((led.get(p) or {}).get('rel_err_mm'), list):
            continue
        say(f"  [follow] {p}: no close read -> {obj} follows the insert-based offsets")
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


NOT_FROM_HOME: set = set()            # parts the startup could not see from home (staged elsewhere)


def derive_sequence_for(obj: str, recorded: list, base_name: str, timeout_s: float = 30.0, samples=None):
    """Build the tool sequence from where the part is actually lying.

    Returns (sequence, note). Falls back to the recorded sequence, unchanged, if the
    part cannot be observed or the planner cannot answer — deriving is an improvement
    on the recording, never a new way to fail.
    """
    # samples from a shared still window (observe_many) when the caller already looked; else observe here
    obs = {'samples': samples} if samples else observe_object_orientation(obj, timeout_s)
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
                            timeout=180, env=make_env(), cwd=str(REPO_ROOT))
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
        # A primitive that exits 0 without its result block crashed after the motion or printed it broken - not a
        # success (AR2). Scripts outside primitives/ (nudge_part.py) report by exit code only and keep that.
        m_ = argv[argv.index('-m') + 1] if '-m' in argv else ''
        if proc.returncode == 0 and m_.startswith('primitives.'):
            msg = f"  [{label}] {m_} exited 0 with no __RESULT_JSON__ block -> failure\n"
            print(msg, end='', flush=True)
            if log_fh:
                log_fh.write(msg)
            return {"result": "failure", "error": "exit 0 with no __RESULT_JSON__ block", "label": label, "returncode": 0}
        return {"result": "failure" if proc.returncode != 0 else "success",
                "error": "no __RESULT_JSON__ block in output" if proc.returncode != 0 else None,
                "label": label, "returncode": proc.returncode}
    rj.setdefault("label", label)
    rj.setdefault("returncode", proc.returncode)
    if ('-m' not in argv and proc.returncode == 0 and rj.get('result') == 'failure' and rj.get('contact')
            and str(rj.get('error', '')).startswith('contact')):
        # a script (not a primitive) echoes its children's JSON: a guarded landing's contact stop is its last block and
        # reads 'failure' by design - judge the script by its exit code (run 33b: a pick-room relocation that had
        # worked failed the run; disassemble.run applies the same rule to its children since 10-19)
        rj = dict(rj, result='success', judged='exit code 0 (last JSON a contact stop)')
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
        self.aim = None                           # seams.aim.Aim of the last insert argv built (drive-mate sends it)

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

from seams.aim import FOLLOWS_SEATED, aim_for_insert, load_facts   # noqa: E402  (AR1: the aim rules live there)
# Written after every successful insert, entry removed when disassemble.py pulls the part out:
# it describes the parts seated NOW, whichever process seated them (a line_green-only retry
# finds the U parts' offsets without any time window).


def record_seat_offset(obj: str, since: float, result=None) -> None:
    """Store where `obj` seated vs the prediction: from the insert's own outcome when it returned one (drive-mate's
    measured_correction, judged at rest - AR2), else from the insert log written since `since`. Says so when neither
    has it (attempt 22: an empty touch CSV recorded nothing, silently, and line_green aimed at the bare base)."""
    try:
        if result is not None and result.get('measured_correction') is not None:
            m = [round(1000 * float(v), 2) for v in result['measured_correction']]
        else:
            from fmb1_loop import insert_summary
            rows = [r for r in insert_summary(since) if r.get('object') == obj and r.get('outcome') == 'success'
                    and r.get('seat_minus_pred_mm')]
            if not rows:
                print(f"  [seated-offsets] {obj}: NOT recorded - the insert returned no outcome and no successful trace "
                      f"since {time.strftime('%H:%M:%S', time.localtime(since))}", flush=True)
                return
            m = list(rows[-1]['seat_minus_pred_mm'])
        # Total offset vs the UNbiased prediction = measured (vs the biased aim) + the bias applied.
        bx, by = _APPLIED_BIAS.get(obj, (0.0, 0.0))
        tot = [round(m[0] + 1000 * bx, 2), round(m[1] + 1000 * by, 2)] + m[2:]
        SEATED.record_seat_offset(obj, tot)
        print(f"  [seated-offsets] {obj}: seat - prediction {tot} mm recorded", flush=True)
    except Exception as e:
        print(f"  [seated-offsets] {obj}: not recorded ({e})")


def forget_seat_offset(obj: str) -> None:
    SEATED.forget(obj, verdict=False)


_APPLIED_BIAS = {}               # part -> (dx, dy) m seat-history bias in this run's insert aim (record_seat_offset)


def _seam_call(args_, timeout_s):
    p = subprocess.Popen(['python3', '-u', '-m', 'seams.client', *args_], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=make_env(), cwd=str(REPO_ROOT))
    try:
        out, err = p.communicate(timeout=timeout_s)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        # let the client cancel its goal (the server then SIGINTs the insert's process group); subprocess.run SIGKILLed
        # it 0.25 s after Ctrl-C, before the cancel went out (AR3 review 10-22)
        p.send_signal(signal.SIGINT)
        try:
            p.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            p.terminate()
        raise
    line = next((l for l in reversed(out.splitlines()) if l.startswith('__SEAM__ ')), None)
    return json.loads(line[len('__SEAM__ '):]) if line else {'error': (err.strip().splitlines() or ['no answer'])[-1]}


def insert_via_drive_mate(obj, base, argv, snap, say):
    """SEAMS_DRIVE=1 (M3a): the insert through /ur5e/drive_mate instead of translate_object directly. Same target and
    grasp the direct call would use, expressed in the seam's terms: target = the base pose the insert aims at (aim
    corrections already applied, the per-part seat z removed - the server adds it), held = the part in the flange
    frame (grasp id + grasp depth ride in it). Result mapped back to the replay's dict."""
    from seams.client import mate_for
    from seams.drive_mate_server import held_pose
    from primitives.shared.config import held_tool_offset

    def after(flag, n):
        i = argv.index(flag); return [float(v) for v in argv[i + 1:i + 1 + n]]
    mate = mate_for(obj, base)
    # the flange the chained held orientation was produced with (not now: after a wrist turn the world orientation is
    # stale - the in-hand yaw grew +1.23 -> +11.15 -> +19.19 deg over three attempts, 2026-10-21)
    tq = getattr(STATE_REF.get('state'), 'held_tcp_quat', None) or (snap.get('tcp') or {}).get('quat_xyzw')
    if mate is None or tq is None:
        return {'result': 'failure', 'error': f'drive_mate: no mate for {obj} on {base}' if mate is None else 'no TCP'}
    aim = getattr(STATE_REF.get('state'), 'aim', None)
    if '--final-base-pos' not in argv or aim is None or aim.obj != obj:   # the seam takes a VisionPose, never a default
        return {'result': 'failure', 'error': f'drive_mate: no observed {base} pose to aim at'}
    # the MEASURED base: drive-mate aligns on it and applies the corrections itself (AR3) - sending the aim would apply
    # them twice. Seam terms: no per-part seat z (the server adds it)
    pos, q = list(aim.measured[0]), list(aim.measured[1])
    gid = int(after('--grasp-id', 1)[0]); held_q = after('--current-object-orientation', 4)
    off_ = float(getattr(STATE_REF.get('state'), 'held_yaw_offset', 0.0) or 0.0)
    if off_:
        held_q = list((R_.from_euler('z', off_, degrees=True) * R_.from_quat(held_q)).as_quat())   # the measured part
    hp, hq = held_pose(obj, gid, held_q, tq, float(np.asarray(held_tool_offset())[2]))
    from seams.aim import corrections_json
    say(f"  [drive_mate] {mate}: measured base ({1000*pos[0]:.1f}, {1000*pos[1]:.1f}) mm, grasp {gid}, corrections "
        f"{[c.name for c in aim.corrections]}")
    out = _seam_call(['drive', '--mate', mate, '--target', *map(str, pos + q), '--held', *map(str, hp + hq),
                      '--corrections', corrections_json(aim)], 1860)
    al_ = out.pop('align', None) or {}
    say(f"  [drive_mate] {mate}: aligned by {al_.get('route')} (log {al_.get('log')}), aim "
        f"{[round(1000 * v, 1) for v in (al_.get('aim') or [[float('nan')] * 3])[0][:2]]} mm")
    say(f"  [drive_mate] {mate}: {json.dumps(out)}")
    if out.get('base_measured') and STATE_REF.get('args') is not None:
        STATE_REF['args'].measured_base = (tuple(out['base_measured'][0]), tuple(out['base_measured'][1]))
    ok = bool(out.get('seated'))
    return dict(out, result='success' if ok else 'failure', movement_type='insert', via='drive_mate',
                error='' if ok else (out.get('error') or out.get('refusal') or f"stage {out.get('stage_reached')}"))


def BASE_FIXED_CFG() -> bool:
    from primitives.shared.config import BASE_FIXED
    return bool(BASE_FIXED)


def one_node() -> bool:
    """Operator 2026-10-21 ('calibrate parts once, then assemble; compliant insertion takes the remaining error as one
    node; nudge is not part of compliant insertion'): with drive-mate inserting (SEAMS_DRIVE=1) the base is read once
    at startup, each part once at its pick, drive-mate's at-rest result decides each insert, nothing touches a part
    after its insert, and ExecuteMate certifies every mate once at the end from one view (report only)."""
    return os.environ.get('SEAMS_DRIVE') == '1'


TRACK_VIEW_Z = 0.56            # flange height over the base for the in-hand read: the startup base view's height, so the
                               # base is beyond the D415's ~0.45 m depth minimum while the held part (~0.2 m) drops out


def track_base_in_hand(base, observed_base, say, log_fh=None, held_obj=None):
    """Free base (2026-10-21 19:31-19:35): two u_brown rim searches turned base1 from 0.19 to ~8.4 deg (8 N lateral
    search force on an untaped base) and the next attempt aimed at the startup read. With the part in hand, fly the
    camera over the base, read it from its corner markers by solvePnP (marker_yaw.base_pose_world: 2.4 px, repeats
    to 0.1 mm / 0.03 deg; the localizer drops frames with a held part in them) and return the base pose to aim at.
    Rule 1: re-read only after something pushed it - every insert attempt touches the base. Returns the new
    ((x, y, z), quat) or the old one when not read."""
    if observed_base is None:
        return observed_base
    from compliant_insertion_studio.wrapper.marker_yaw import base_pose_world
    from seat_check import X_CAM
    snap = snapshot_robot_state(); tcp = snap.get('tcp') or {}
    if not tcp.get('quat_xyzw'):
        return observed_base
    (bx, by, _), _ = observed_base
    Rt = R_.from_quat(tcp['quat_xyzw']).as_matrix()
    cam = Rt @ np.asarray(X_CAM)[:3, 3]                            # camera offset from the flange, world frame
    tx, ty = bx - cam[0], by - cam[1]                               # camera straight over the base centre
    r = _move_unless_there((tx, ty, TRACK_VIEW_Z), list(tcp['quat_xyzw']), 'base track view (part in hand)', log_fh)
    if r.get('result') == 'failure':
        say(f"  [base-track] view not reached ({r.get('error', '')}) - aiming at the previous read"); return observed_base
    # The pose comes from the 6-dof CAD fit to the depth cloud (operator 10-21: depth + CAD carry far more than the
    # markers; the localizer/markers only say which object is roughly where = the seed). Markers seed it.
    seed, info = base_pose_world(base)
    if seed is None:
        seed, info = observed_base, dict(info, seed='previous read')
    try:
        from base_icp import read_icp
        # the parts seated before the held one are part of the rigid body being read (10-22 06:48: line_green aims at
        # the U openings, which are where the U parts physically seated; the bare-base read is several mm off them -
        # three attempts rode the U tops at +61 mm). They add tall, distinct geometry to the fit too.
        seated_ = ()
        if held_obj is not None:
            try:
                from compliant_insertion_studio.wrapper.cad_geometry import get_seated_so_far
                seated_ = tuple(get_seated_so_far(base, held_obj))
            except Exception:
                seated_ = ()
        pose, icp = read_icp(base, seed, seated=seated_)
        if seated_:
            icp = dict(icp, seated=list(seated_))
    except Exception as e:
        pose, icp = None, {'error': str(e)}
    info = dict(info, icp=icp)
    if pose is None:
        say(f"  [base-track] {base}: CAD fit failed in hand ({icp}) - aiming at the previous read"); return observed_base
    (ox, oy, oz), oq = observed_base
    dyaw = float(np.degrees((R_.from_quat(pose[1]) * R_.from_quat(oq).inv()).as_rotvec()[2]))
    dxy = 1000 * np.hypot(pose[0][0] - ox, pose[0][1] - oy)
    say(f"  [base-track] {base} from the CAD fit to depth (part in hand): moved {dxy:.1f} mm, turned {dyaw:+.2f} deg "
        f"since the aim {json.dumps(info, default=str)}")
    # keep the measured z of the CAD fit (markers sit on the top face; their z carries the depth of the top plane)
    yaw_only = R_.from_euler('z', float(R_.from_quat(pose[1]).as_euler('xyz')[2]))
    tilt = R_.from_quat(oq) * R_.from_euler('z', float(R_.from_quat(oq).as_euler('xyz')[2])).inv()
    return ((pose[0][0], pose[0][1], oz), tuple(float(v) for v in (tilt * yaw_only).as_quat()))


HOVER_ALIGN_ABOVE_M = 0.10      # TCP height over its predicted seat for the in-image part-vs-socket alignment
HOVER_ALIGN_TOL_M = 0.001
HOVER_ALIGN_ITERS = 3           # reads (median)
HOVER_ALIGN_MAX_M = 0.012       # a larger 'offset' is a misread, not a pick error


def hover_align(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Operator 10-21 ("do you see how much offset there is ... it is not even above the socket"): the held part was
    ~5-10 mm off its socket while the chained pose put it there - the in-hand error of the pick. Over the predicted
    seat, the camera sees the part's top face against the red base top: marker_yaw.held_offset measures the part's
    real position against the chain in one still frame (edges vs the projected CAD outline, base from the same frame),
    the arm moves by minus that, and measures again (<= 3 rounds). Returns the base pose to aim the insert at:
    shifted by minus the total correction, so the seat target moves with the measured part."""
    if observed_base is None or held_q is None:
        return observed_base
    from compliant_insertion_studio.wrapper.marker_yaw import held_offset
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset, held_tool_offset_for
    snap = snapshot_robot_state(); tcp = snap.get('tcp') or {}
    if not tcp.get('quat_xyzw'):
        return observed_base
    (bx, by, bz), bq = observed_base
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=tcp['quat_xyzw'],
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); sq = pred['predicted_tcp_at_seat']['quat_xyzw']
    # The measured offset is the part's error IN THE HAND (real - chained, relative to the flange): moving the arm does
    # not change it (10-21: four rounds read (1.0,-4.8) (1.2,-6.0) (1.2,-4.5) (1.4,-6.3) mm). Measure it a few times
    # over the seat, take the median, and move the aim by minus it once.
    tgt = [st[0], st[1], st[2] + HOVER_ALIGN_ABOVE_M]
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                        *[f"{v:.5f}" for v in tgt], '--quat', *[f"{v:.6f}" for v in sq], '--min-z', '0.15'],
                       label='hover align view', timeout_s=60, log_fh=log_fh)
    if r.get('result') == 'failure':
        say(f"  [hover-align] {obj}: hover not reached ({r.get('error', '')})"); return observed_base
    reads = []
    _emitter(False)
    try:
        for k in range(HOVER_ALIGN_ITERS):
            d, info = held_offset(obj, base, held_q, grasp_id)
            if d is None:
                say(f"  [hover-align] {obj}: read {k} not measured ({info.get('error')})"); continue
            reads.append(np.asarray(d[:2]))
            say(f"  [hover-align] {obj} read {k}: part {info['dxy_mm']} mm off its chained pose "
                f"({info['n_edge']} edge samples, resid {info['resid_px']} px)")
    finally:
        _emitter(True)
    if len(reads) < 2:
        return observed_base
    total = np.median(np.array(reads), axis=0)
    if np.hypot(*total) > HOVER_ALIGN_MAX_M:
        say(f"  [hover-align] {obj}: {1000*np.hypot(*total):.1f} mm is more than a pick can be off - not applied")
        return observed_base
    say(f"  [hover-align] {obj}: aim corrected by ({-1000*total[0]:+.1f}, {-1000*total[1]:+.1f}) mm (median of {len(reads)})")
    return ((bx - total[0], by - total[1], bz), bq)


def _footprints_overlap(base, a, b, margin_m=0.005):
    """Do the seated CAD footprints (base-frame xy boxes) of parts a and b overlap, within margin_m?"""
    from grasp_depth import load_mesh, surface_points
    from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
    box = []
    for o in (a, b):
        sp_, sq_ = load_seat_pose_in_base(base, o); V_, F_ = load_mesh(o)
        P_ = R_.from_quat(sq_).apply(surface_points(V_, F_, 0.003)) + sp_
        box.append((P_[:, 0].min(), P_[:, 0].max(), P_[:, 1].min(), P_[:, 1].max()))
    (ax0, ax1, ay0, ay1), (bx0, bx1, by0, by1) = box
    return ax0 < bx1 + margin_m and bx0 < ax1 + margin_m and ay0 < by1 + margin_m and by0 < ay1 + margin_m


def seated_under(base, obj):
    """Parts seated before obj whose CAD footprint overlaps obj's: it threads over them first (a corridor stage - yellow
    over line_green) rather than meeting the base rim (cad_geometry.seated_under, shared with touch_insert)."""
    from compliant_insertion_studio.wrapper.cad_geometry import seated_under as _su
    return _su(base, obj)


DSERVO_SPREAD_YAW_PX = 20.0     # a yaw slopes the far edge and spreads the fit: up to this the yaw is turned out first
DSERVO_ITERS = 6


HELD_AXIAL = os.environ.get('HELD_AXIAL', '1') == '1'
HELD_AXIAL_MAX_M = 0.006         # a larger 'offset' is a misread (the play along line_green's pocket is +-1.4 mm)
HELD_AXIAL_MAD_PX = 3.0


def held_axial_correction(obj, grasp_id, held_q, outline, tcp_pose, say):
    """World xy (m) to move the aim so the held part's CAD grasp point lands on the seat: minus the part's offset along
    image-y (the far edge vs the CAD part held at its grasp, projected from the TCP through the hand-eye). None when the
    far edge does not fit (MAD) or the offset is implausible."""
    try:
        import cv2
        from compliant_insertion_studio.wrapper.marker_yaw import _intrinsics
        from compliant_insertion_studio.wrapper.cad_lookup import load_grasp_offset_in_object
        from seat_check import X_CAM
        from handeye_icp_survey import T as _T
        from grasp_depth import load_mesh, surface_points
        from primitives.shared.config import held_tool_offset_for
        tp, tq = np.asarray(tcp_pose[0], float), np.asarray(tcp_pose[1], float)        # the TCP the outline was taken at
        Rh = R_.from_quat(held_q)
        centre = tp + R_.from_quat(tq).apply([0, 0, float(held_tool_offset_for(obj)[2])])
        P = Rh.apply(surface_points(*load_mesh(obj), 0.0015)) + centre - Rh.apply(load_grasp_offset_in_object(obj, int(grasp_id)))
        K, D = _intrinsics(); Twc = _T(tp, tq) @ X_CAM; Tcw = np.linalg.inv(Twc)
        Pc = (Tcw[:3, :3] @ P.T).T + Tcw[:3, 3]
        uv, _ = cv2.projectPoints(Pc.astype(np.float32), np.zeros(3), np.zeros(3), K, D); uv = uv.reshape(-1, 2)
        dy, zs = [], []
        for x, y in outline['far']:
            m = np.abs(uv[:, 0] - x) < 2.5
            if m.sum() >= 3:
                i = np.argmin(np.where(m, uv[:, 1], np.inf)); dy.append(y - uv[i, 1]); zs.append(Pc[i, 2])
        if len(dy) < 8:
            say(f"  [held-axial] {obj}: far edge out of the projection ({len(dy)} columns)"); return None
        dy = np.asarray(dy); med = float(np.median(dy)); mad = float(1.4826 * np.median(np.abs(dy - med)))
        Z = float(np.median(zs))
        # the left edge (image-x: across a bar) the same way: runs 37/38b/39 read -2.1 / -1.9 / -2.2 mm (MAD 0.1-0.2 px)
        # where the probe lattice then found the pocket 2.0-2.5 mm across (50 probes in run 39)
        dx = []
        for x, y in outline.get('left') or []:
            m = np.abs(uv[:, 1] - y) < 2.5
            if m.sum() >= 3:
                i = np.argmin(np.where(m, uv[:, 0], np.inf)); dx.append(x - uv[i, 0])
        dx = np.asarray(dx); medx = float(np.median(dx)) if len(dx) >= 5 else 0.0
        madx = float(1.4826 * np.median(np.abs(dx - medx))) if len(dx) >= 5 else float('inf')
        if madx > HELD_AXIAL_MAD_PX:
            medx = 0.0                                           # the left edge does not fit: along the far edge only
        d = Twc[:3, :3] @ np.array([medx * Z / K[0, 0], med * Z / K[1, 1], 0.0])     # part - CAD grasp, world
        say(f"  [held-axial] {obj}: far edge {med:+.1f} px (MAD {mad:.1f}, {len(dy)} columns), left edge {medx:+.1f} px "
            f"(MAD {madx:.1f}, {len(dx)} rows) from the CAD grasp = ({1000*d[0]:+.1f}, {1000*d[1]:+.1f}) mm")
        if mad > HELD_AXIAL_MAD_PX or np.hypot(d[0], d[1]) > HELD_AXIAL_MAX_M:
            say(f"  [held-axial] {obj}: not used (MAD > {HELD_AXIAL_MAD_PX} px or > {1000*HELD_AXIAL_MAX_M:.0f} mm)"); return None
        return np.array([-d[0], -d[1]])
    except Exception as e:
        say(f"  [held-axial] {obj}: not measured ({e})"); return None


DSERVO_ROBUST = os.environ.get('DSERVO_ROBUST', '1') == '1'   # far edge by its median when only its std is spread (38b: converged, seated)


def depth_servo(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Observe -> move -> observe with the target in the SAME frame as the image (10-22 11:16, the yellow seated by it):
    (1) wrist to the base yaw, (2) the held part's outline over bare table, (3) base1 fitted to depth with the camera
    centred over it (track_base_in_hand, seated parts in the fit), (4) the TCP translated straight over the predicted
    seat at that height - the hand-eye error cancels but for its rotation x the ~5 cm move (~0.4 mm) - then rounds:
    socket_shift_world; a yaw over 0.4 deg is turned out first (it alone spread the yellow's fit to 12 px: 2 px after),
    then xy steps by -d until < 1 mm. Returns (base pose shifted so its seat lies under the aim, held quat) or None when
    it cannot measure (outline, base read or fit) - the caller keeps its own aim."""
    if observed_base is None or held_q is None:
        return None
    from compliant_insertion_studio.wrapper.marker_yaw import socket_shift_world, held_outline_px, still_frame, _tcp_pose_now
    from compliant_insertion_studio.wrapper.align_at_hover import rel_yaw_error, turn_wrist
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset_for
    (bx, by, bz), bq = observed_base
    err, _t = rel_yaw_error(obj, base, R_.from_quat(bq), R_.from_quat(held_q))
    if abs(err) > 0.3 and turn_wrist(err):
        held_q = list((R_.from_euler('z', err, degrees=True) * R_.from_quat(held_q)).as_quat())
        say(f"  [dservo] {obj}: wrist turned {err:+.2f} deg to the base yaw")
    q = (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw')
    r = _move_unless_there((bx + 0.17, by, 0.32), list(q), 'dservo: outline view over table', log_fh)   # in process
    _emitter(False)
    try:
        img_, _ = still_frame(n_avg=4)
    finally:
        _emitter(True)
    outline = held_outline_px(img_) if (img_ is not None and r.get('result') != 'failure') else None
    if outline is None:
        say(f"  [dservo] {obj}: held part not outlined over the table"); return None
    stem = REPO_ROOT / 'compliant_insertion_studio' / 'logs' / f"dservo_{obj}_{time.strftime('%Y%m%d_%H%M%S')}"
    tp_outline, hq_outline = _tcp_pose_now(), list(held_q)           # the outline's frame (the base read moves the arm)
    try:
        # the held part over bare table, every grasp: where it sits in the hand (camera rigid with the gripper) - the
        # data for measuring a bar's axial grasp error without the base (line_green: 2-4 mm toward -y, 16/19 inserts)
        import cv2 as _cv2
        json.dump({'outline': outline, 'tcp': [list(map(float, v)) for v in tp_outline], 'held_q': list(held_q)},
                  open(f"{stem}_outline.json", 'w')); _cv2.imwrite(f"{stem}_table.png", img_)
    except Exception:
        pass
    ob = track_base_in_hand(base, observed_base, say, log_fh, held_obj=obj)   # depth fit, camera over the base
    if ob is not None and ob is not observed_base:
        LAST_TRACK_READ['pose'] = ob
    _emitter_hold()                                                   # the rounds are colour-only
    if ob is None or ob is observed_base:
        say(f"  [dservo] {obj}: base not re-read in hand")
        # no base in this frame - the hand still is: the held part's far edge over bare table against the CAD grasp
        # (camera rigid with the gripper) gives where it sits along image-y (a bar's axis). 10-23: line_green's far end
        # read 18.3 px (MAD 1.1) past a centred grasp = 2.8 mm toward base -y, the overhang that wedged 16 of 19 inserts
        c = held_axial_correction(obj, grasp_id, held_q, outline, tp_outline, say) if HELD_AXIAL else None
        if c is None:
            return None
        (bx_, by_, bz_), bq_ = observed_base
        say(f"  [dservo] {obj}: aim shifted ({1000*c[0]:+.1f}, {1000*c[1]:+.1f}) mm by the held part's place in the hand")
        return ((bx_ + c[0], by_ + c[1], bz_), bq_), held_q
    (bx, by, bz), bq = ob
    off = float(held_tool_offset_for(obj)[2])

    def seat_tcp(q_ee, hq):
        pr = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                                 base_world_quat_xyzw=list(bq), held_quat_xyzw=list(hq), ee_orientation_xyzw=list(q_ee),
                                 flange_offset_m=off)
        return np.array(pr['predicted_tcp_at_seat']['xyz_m'])
    z = TRACK_VIEW_Z
    xy = seat_tcp(_tcp_pose_now()[1], held_q)[:2].copy()
    yaw_total, rounds, ok, misread = 0.0, [], False, False
    for k in range(DSERVO_ITERS):
        q_now = (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw')      # keep the orientation, in process
        rr_ = (_move_unless_there((xy[0], xy[1], z), q_now, f'dservo round {k}: over the aim', log_fh) if q_now else
               run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{xy[0]:.5f}",
                               f"{xy[1]:.5f}", f"{z:.5f}", '--keep-orientation', '--min-z', '0.15'],
                              label=f'dservo round {k}: over the aim', timeout_s=60, log_fh=log_fh))
        if rr_.get('result') == 'failure' or _SAFETY_STOP:
            break
        _emitter(False)
        try:
            img_, _ = still_frame(n_avg=4); tp = _tcp_pose_now(); tp = (np.asarray(tp[0]), np.asarray(tp[1]))
        finally:
            _emitter(True)
        st = seat_tcp(tp[1], held_q)
        d, info = socket_shift_world(obj, base, ob, float(tp[0][2] - st[2]), outline, img=img_, tcp_pose=tp,
                                     debug_path=f"{stem}_r{k}.png")
        rounds.append({'k': k, 'tcp': [tp[0].tolist(), tp[1].tolist()], 'seat_tcp': st.tolist(), 'info': info})
        say(f"  [dservo] {obj} round {k}: {info.get('dxy_mm')} mm, yaw {info.get('yaw_img_deg')} deg, spread "
            f"{info.get('spread_px')} px" + (f" ({info.get('error')})" if d is None else ""))
        if d is not None and np.hypot(*d) > SERVO_MAX_M:
            # more than the whole correction may be (the end check) - a misread too; run 52 stepped 73 mm on it, twice
            misread = True
            say(f"  [dservo] {obj}: reading {1000*np.hypot(*d):.1f} mm is past the servo's {1000*SERVO_MAX_M:.0f} mm "
                f"- a misread fit, its readings not used")
            break
        if d is not None and abs(info.get('yaw_img_deg') or 0.0) > SERVO_MAX_YAW_DEG:
            # a yaw past what the whole servo may turn, read right after the wrist was turned to the base yaw, is the
            # fit matching something else: line_green read 35.9 / 46.1 deg (runs 51 / 49), its far-edge median then aimed
            # 2.5 mm off the held part's measured place along x (clean fits agree with it to 0.1 mm, run 47) and run 51
            # wedged an end +6.7 mm. None of its readings are used.
            misread = True
            say(f"  [dservo] {obj}: image yaw {info.get('yaw_img_deg')} deg is past the servo's {SERVO_MAX_YAW_DEG:.0f} deg "
                f"- a misread fit, its readings not used")
            break
        far_bad = d is not None and info['spread_px'][1] > DSERVO_SPREAD_YAW_PX
        if (far_bad and DSERVO_ROBUST and (info.get('spread_mad_px') or [99, 99])[1] <= SERVO_MAX_SPREAD_PX):
            # the far edge's columns disagree in std but agree in MAD: its median holds, its slope (the yaw) does not -
            # step along the far edge's component only (10-23 line_green: std 101.7 px, median -14.4 px = 2.25 mm along
            # the bar toward -y, the overhang that wedged 16 of 19 inserts)
            d = np.asarray(info['d_far_only_mm'], float) / 1000.0
            say(f"  [dservo] {obj}: far edge std {info['spread_px'][1]} px but MAD {info['spread_mad_px'][1]} px - its "
                f"median only: {info['d_far_only_mm']} mm")
            if np.hypot(*d) <= SERVO_TOL_M:
                xy = xy - d; ok = True; break
            xy = xy - d
            continue
        if d is None or far_bad:
            break                                                    # the far edge decides; a spread left edge is handled below
        yi = info.get('yaw_img_deg') or 0.0
        if abs(yi) > SERVO_YAW_MIN_TURN_DEG:
            t = float(np.clip(yi / SERVO_YAW_GAIN, -5.0, 5.0))
            if abs(yaw_total + t) > SERVO_MAX_YAW_DEG:
                say(f"  [dservo] {obj}: yaw {yaw_total + t:+.1f} deg in all is more than the reads can be off - stopping"); break
            if turn_wrist(t):
                yaw_total += t
                held_q = list((R_.from_euler('z', t, degrees=True) * R_.from_quat(held_q)).as_quat())
                say(f"  [dservo] {obj}: wrist turned {t:+.2f} deg (image yaw)")
                continue
        if info['spread_px'][1] > SERVO_MAX_SPREAD_PX:
            say(f"  [dservo] {obj}: far edge spreads {info['spread_px']} px after the yaw - no xy step"); break
        if info['spread_px'][0] > SERVO_MAX_SPREAD_PX:
            # the far edge fits, the left one does not (13:13 yellow: left 45.5 px, far 1.2 px, far reading 7.8 then
            # 8.3 mm): step along the far edge's component only, as the marker servo does
            d = np.asarray(info['d_far_only_mm'], float) / 1000.0
            say(f"  [dservo] {obj}: left edge spreads {info['spread_px'][0]} px - far-edge component only "
                f"{info['d_far_only_mm']} mm")
        if np.hypot(*d) <= SERVO_TOL_M:
            xy = xy - np.asarray(d); ok = True; break
        xy = xy - np.asarray(d)
    try:
        json.dump({'rounds': rounds, 'held_q': list(held_q), 'base': [list(ob[0]), list(ob[1])], 'aim': list(xy), 'ok': ok},
                  open(f"{stem}.json", 'w'), indent=1)
    except Exception:
        pass
    if not ok and misread and HELD_AXIAL:
        # the held part's place in the hand, from its outline over bare table (taken above), against this frame's base read
        c = held_axial_correction(obj, grasp_id, hq_outline, outline, tp_outline, say)
        if c is not None:
            say(f"  [dservo] {obj}: aim shifted ({1000*c[0]:+.1f}, {1000*c[1]:+.1f}) mm by the held part's place in the hand")
            return ((bx + c[0], by + c[1], bz), bq), held_q
    if not ok:
        say(f"  [dservo] {obj}: not converged - caller keeps its aim"); return None
    st = seat_tcp((snapshot_robot_state().get('tcp') or {}).get('quat_xyzw'), held_q)
    c = np.asarray(xy) - st[:2]
    if np.hypot(*c) > SERVO_MAX_M:
        say(f"  [dservo] {obj}: correction {1000*np.hypot(*c):.1f} mm is more than the reads can be off - not applied"); return None
    say(f"  [dservo] {obj}: converged, aim {1000*c[0]:+.1f} / {1000*c[1]:+.1f} mm from the predicted seat")
    return ((bx + c[0], by + c[1], bz), bq), held_q


def close_base_read(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Base pose from its markers in one frame over obj's seat at the clear height (tilt from the chain, yaw searched,
    translation solved: marker_yaw.base_pose_same_frame). Returns the new ((x, y, z), quat) (z kept) or the old one."""
    if observed_base is None or held_q is None:
        return observed_base
    from compliant_insertion_studio.wrapper.marker_yaw import base_pose_same_frame, still_frame
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset_for
    (bx, by, bz), bq = observed_base
    tcp = (snapshot_robot_state().get('tcp') or {})
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=tcp['quat_xyzw'],
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); q = tcp['quat_xyzw']
    z_view = float(st[2]) + HOVER_ALIGN_ABOVE_M
    cur = tcp.get('xyz')
    if cur and cur[2] < z_view - 0.002:
        run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{cur[0]:.5f}",
                        f"{cur[1]:.5f}", f"{z_view:.5f}", '--quat', *[f"{v:.6f}" for v in q], '--min-z', '0.15'],
                       label='close base read: up', timeout_s=60, log_fh=log_fh)
    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{st[0]:.5f}",
                        f"{st[1]:.5f}", f"{z_view:.5f}", '--quat', *[f"{v:.6f}" for v in q], '--min-z', '0.15'],
                       label='close base read view', timeout_s=60, log_fh=log_fh)
    if r.get('result') == 'failure' or _SAFETY_STOP:
        return observed_base
    _emitter(False)
    try:
        img_, _ = still_frame(n_avg=6); tc = (snapshot_robot_state().get('tcp') or {})
        pose, info = base_pose_same_frame(base, img_, (np.asarray(tc['xyz']), np.asarray(tc['quat_xyzw'])), bq) \
            if img_ is not None else (None, {'error': 'no frame'})
    finally:
        _emitter(True)
    if pose is None or info.get('reproj_px', 99) > 3.0:
        say(f"  [close-base] {base}: not read over {obj}'s seat ({info}) - aiming at the base read")
        return observed_base
    d_mm = 1000 * np.hypot(pose[0][0] - bx, pose[0][1] - by)
    if d_mm > 15.0:
        say(f"  [close-base] {base}: close read {d_mm:.1f} mm from the base read - more than the read can be off, not used")
        return observed_base
    say(f"  [close-base] {base} over {obj}'s seat: {d_mm:.1f} mm from the 0.56 m read ({json.dumps(info)})")
    return ((pose[0][0], pose[0][1], bz), tuple(pose[1]))


SERVO_MEASURED: dict = {}         # obj -> did the last hover_servo take a reading


LAST_TRACK_READ = {}            # the hover servo's in-hand base read (with the turned wrist)


def align_for_insert(obj, base, measured_base, held_quat, grasp_id, say, log_fh=None):
    """Free base, one-node: bring the held part over its seat before the insert (AR3: the call drive-mate's align step
    makes). From the last MEASURED base: the in-hand base re-read, then by route - a prismatic part turns to the base
    yaw and re-reads the base close (it searches itself); a part threading over seated parts (CAD) runs the depth-frame
    servo (close read as fallback); any other the marker servo (depth-frame servo when it took no reading).
    Returns {aim_base, measured_base, held_quat, tcp_quat (the flange after a wrist turn, else None), route}."""
    from compliant_insertion_studio.wrapper.align_at_hover import rel_yaw_error, turn_wrist
    out = dict(aim_base=None, measured_base=None, held_quat=held_quat, tcp_quat=None, route='')
    try:
        from primitives.translate_object import PRISMATIC_ROUTE_OBJECTS as _PRISM
    except Exception:
        _PRISM = ()
    mode = os.environ.get('HOVER_ALIGN', 'servo')
    servo_route = mode == 'servo' and obj not in _PRISM
    if servo_route:
        # the marker servo and the depth servo each re-read the base in hand themselves (the read they project with): a
        # read here first only fed the wrist turn before it (run 34: ~15 s per U part). The depth servo's took the
        # yellow base view -> outline over the table (back by the pick) -> base view again: 11.5 s and a backtrack
        # (run 50b; the two reads agreed to 0.4 mm / 0.04 deg, the startup read's yaw to 0.18 deg)
        ob = measured_base
        LAST_TRACK_READ.pop('pose', None)
    else:
        ob = track_base_in_hand(base, measured_base, say, log_fh, held_obj=obj)
    out['measured_base'] = ob

    def _tcp_q():
        return (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw')
    if mode == 'servo' and obj in _PRISM:
        # the prismatic route searches for itself; the servo aim (a +-2.5 mm dip grid needs it) only misleads it:
        # 10-22 05:17 line_green's outline fit broke (140 px), its 'best reading' moved the aim 6 mm and the bar stopped
        # 60 mm high, where it had seated from the read aim before (slot 6-8 mm away, found by the route itself)
        out['route'] = 'prismatic: yaw + close read'
        say(f"  [servo] {obj}: prismatic route - it searches itself, no xy servo")
        # but the wrist still turns to the base's yaw first: skipping the whole servo (9a19b76) skipped this turn and
        # line_green arrived +13.2 deg off its slot (base1 at ~16 deg; the U openings it threads need < 1.3 deg) -
        # jammed on the U tops at +61 mm, 10-22 06:27 (INSERT_GEOMETRY: the corridor stage is a yaw wedge)
        try:
            if ob is not None and held_quat is not None:
                err_, _t = rel_yaw_error(obj, base, R_.from_quat(ob[1]), R_.from_quat(held_quat))
                if abs(err_) > 0.3 and turn_wrist(err_):
                    held_quat = list((R_.from_euler('z', err_, degrees=True) * R_.from_quat(held_quat)).as_quat())
                    out['tcp_quat'] = _tcp_q()
                    say(f"  [servo] {obj}: wrist turned {err_:+.2f} deg to the base yaw")
        except Exception as e_:
            say(f"  [servo] {obj}: yaw to base not applied ({e_})")
        # and the base is re-read from its markers in a CLOSE frame over the seat (no xy servo): the 0.56 m CAD
        # fit carries the hand-eye rotation error x ~0.5 m of lever (several mm); line_green needs +-1.1 mm across
        # the U openings. Bare and seated-assembly fits agreed to 0.4 mm (10-22 07:00), so the U parts are not the
        # offset - the read height is. From the clear height (~0.17 m over the base top) the lever is ~1/3.
        try:
            ob = close_base_read(obj, base, ob, held_quat, grasp_id, say, log_fh)
            out['measured_base'] = ob                                # a measurement too (markers, one close frame)
            # the wrist turned to the base yaw BEFORE this read: turn again to the closer read's yaw if they differ
            if ob is not None and held_quat is not None:
                err2_, _t2 = rel_yaw_error(obj, base, R_.from_quat(ob[1]), R_.from_quat(held_quat))
                if abs(err2_) > 0.3 and turn_wrist(err2_):
                    held_quat = list((R_.from_euler('z', err2_, degrees=True) * R_.from_quat(held_quat)).as_quat())
                    out['tcp_quat'] = _tcp_q()
                    say(f"  [servo] {obj}: wrist turned {err2_:+.2f} deg to the close read's base yaw")
        except Exception as e_:
            say(f"  [servo] {obj}: close base read not applied ({e_})")
    elif mode == 'servo' and seated_under(base, obj):
        # threads over seated parts first (CAD): the depth-frame servo (10-22 11:16 seated the yellow), the close
        # marker read as the fallback when it cannot measure
        out['route'] = 'depth servo'
        try:
            ds = depth_servo(obj, base, ob, held_quat, grasp_id, say, log_fh)
        finally:
            _emitter_release()
        if LAST_TRACK_READ.get('pose') is not None:
            out['measured_base'] = LAST_TRACK_READ['pose']           # the depth servo's in-hand read
        if ds is not None:
            ob, held_quat = ds
        else:
            try:
                ob = close_base_read(obj, base, ob, held_quat, grasp_id, say, log_fh)
            except Exception as e_:
                say(f"  [dservo] {obj}: close base read not applied ({e_})")
        out['tcp_quat'] = _tcp_q()
    elif mode == 'servo':
        # the marker servo first (u_brown seated first try with it in attempts 16-18; the depth servo's clean,
        # converged reading put u_brown 6 mm off at 12:39 - not yet understood), the depth-frame servo when the
        # marker servo took no reading (12:05/12:22 u_orange: 0-1 base markers over its far-side seat; the depth
        # servo converged and it entered on the lattice)
        out['route'] = 'marker servo'
        ob0_, hq0_ = ob, held_quat
        try:
            ob, hq_ = hover_servo(obj, base, ob0_, hq0_, grasp_id, say, log_fh)
        finally:
            _emitter_release()
        if LAST_TRACK_READ.get('pose') is not None:
            out['measured_base'] = LAST_TRACK_READ['pose']           # the servo's in-hand read: drive-mate's measured base
        if not SERVO_MEASURED.get(obj) and not _SAFETY_STOP:
            out['route'] = 'marker servo -> depth servo'
            say(f"  [servo] {obj}: no marker reading - the depth-frame servo")
            try:
                ds = depth_servo(obj, base, ob0_, hq_ if hq_ is not None else hq0_, grasp_id, say, log_fh)
            finally:
                _emitter_release()
            if ds is not None:
                ob, hq_ = ds
        if hq_ is not None and held_quat is not None and hq_ is not held_quat:
            held_quat = hq_                                          # the wrist turned: the chained held pose turns with it
            out['tcp_quat'] = _tcp_q()
    elif mode == '1':
        out['route'] = 'hover align'
        ob = hover_align(obj, base, ob, held_quat, grasp_id, say, log_fh)
    out['aim_base'], out['held_quat'] = ob, held_quat
    return out


GRASP_DEPTH_MEAS_MIN_M = 0.003   # a servo touchdown this far over the CAD rim is the grasp depth, not noise ...
GRASP_DEPTH_MEAS_MAX_M = 0.012   # ... and up to this (more is a part on something else - the yellow met line_green's top +18 mm)
GRASP_DEPTH_FIX = {}             # obj -> depth correction (m) measured this grasp; drive_align reports it to drive-mate


SERVO_INPROC = os.environ.get('SERVO_INPROC', '1') == '1'   # servo moves in process (benched empty 10-23: 2.1 s a move vs 4.6)
SERVO_SHORT_M, SERVO_SHORT_SPEED_M_S, SERVO_SHORT_MIN_S = 0.020, 0.015, 1.0   # in process: <= 20 mm timed by distance


def _move_unless_there(xyz, q, label, log_fh=None, extra=()):
    """A servo move, skipped when the arm is already there (within 0.3 mm, same orientation): the 10-22 motion trace
    counted 15 servo moves to exactly the previous target (hover from the clear height = clear, up for a turn, done) at
    ~4 s each."""
    tcp = snapshot_robot_state().get('tcp') or {}
    if (tcp.get('xyz') and tcp.get('quat_xyzw') and np.linalg.norm(np.asarray(tcp['xyz']) - np.asarray(xyz, float)) < 0.0003
            and abs(float(np.dot(tcp['quat_xyzw'], q))) > 0.99999):
        return {'result': 'success', 'skipped': 'already there'}
    kw = {}
    ex = list(extra)
    while ex[:1] and ex[0] in ('--max-force', '--min-duration') and len(ex) >= 2:   # the guarded hover: same guard in process
        kw['max_force' if ex[0] == '--max-force' else 'min_duration'] = float(ex[1]); ex = ex[2:]
    if SERVO_INPROC and not ex:
        # the same plan_move the CLI runs, on one node, released before the next camera read (which starts its own
        # context): run 33's servo moves were ~4 s each, ~1.5 s of it a process start (u_brown: 8 moves / 99 s align)
        from compliant_insertion_studio.wrapper.touch_insert import move as _tmove, close_node as _close
        t0 = time.time()
        d_ = (float(np.linalg.norm(np.asarray(tcp['xyz']) - np.asarray(xyz, float)))
              if tcp.get('xyz') and abs(float(np.dot(tcp.get('quat_xyzw') or [0, 0, 0, 0], q))) > 0.9999 else 1.0)
        try:
            # short moves (the servo's 3-15 mm steps) timed by distance, as the touch probes' are: the planner's floor
            # held them at ~2.5 s (bench, nothing gripped: 3.5 s a move in process vs 4.6 s by subprocess)
            short = d_ <= SERVO_SHORT_M and 'min_duration' not in kw
            r = _tmove(list(map(float, xyz)), list(map(float, q)), min_z=0.15,
                       duration=max(SERVO_SHORT_MIN_S, d_ / SERVO_SHORT_SPEED_M_S) if short else None, **kw)
        finally:
            _close()
        fh = _TIMING.get('fh')
        if fh is not None:
            fh.write(json.dumps({'kind': 'move_inproc', 'label': label, 't0': t0, 't1': time.time(),
                                 'result': r.get('result')}) + '\n')
        return r
    return run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                           *[f"{float(v):.5f}" for v in xyz], '--quat', *[f"{v:.6f}" for v in q], '--min-z', '0.15',
                           *extra], label=label, timeout_s=60, log_fh=log_fh)


SEAT_CHECK_FIRST = os.environ.get('SEAT_CHECK_FIRST', '0') == '1'
SERVO_PRIOR = os.environ.get('SERVO_PRIOR', '1') == '1'
SERVO_PRIOR_LOG = REPO_ROOT / 'ablations' / 'logs' / 'servo_priors.jsonl'
SERVO_PRIOR_MAD_M, SERVO_PRIOR_MAD_DEG = 0.001, 1.0


def servo_prior(obj):
    """The marker servo's measured correction for obj, in the BASE frame (so it holds at any base yaw): the median of its
    last 6 converged servos (>= 3, agreeing within 1 mm / 1 deg MAD), or None. u_brown's servo moved its aim (-3.3..-4.5,
    -6.7..-8.2) mm and turned -1..-4.5 deg in every run of 10-22/23 - round 0 always found it ~9 mm / ~2 deg off, and
    each round is a lift, a turn, a step and a lowering (~13 s): the zigzag the operator saw. The servo still measures;
    the prior only starts it nearer, so a pose where the prior is wrong costs the round it costs today."""
    try:
        rows = [json.loads(l) for l in open(SERVO_PRIOR_LOG) if l.strip()]
    except Exception:
        return None
    rows = [r for r in rows if r.get('obj') == obj][-6:]
    if len(rows) < 3:
        return None
    d = np.array([r['dxy_base_m'] for r in rows], float); y = np.array([r['yaw_deg'] for r in rows], float)
    md, my = np.median(d, 0), float(np.median(y))
    if (np.median(np.abs(d - md), 0) > SERVO_PRIOR_MAD_M).any() or np.median(np.abs(y - my)) > SERVO_PRIOR_MAD_DEG:
        return None
    return md, my, len(rows)


def _record_servo(obj, c_world, yaw_deg, base_q, rounds):
    try:
        yb = float(R_.from_quat(base_q).as_euler('xyz')[2])
        cb = R_.from_euler('z', -yb).apply([c_world[0], c_world[1], 0.0])[:2]
        with open(SERVO_PRIOR_LOG, 'a') as f:
            f.write(json.dumps({'obj': obj, 't': time.time(), 'dxy_base_m': [float(cb[0]), float(cb[1])],
                                'yaw_deg': float(yaw_deg or 0.0), 'rounds': int(rounds)}) + '\n')
    except Exception:
        pass


def hover_servo(obj, base, observed_base, held_q, grasp_id, say, log_fh=None):
    """Operator 10-21 ("the yaw fix seemed to work but not the offset"): align the held part to its SOCKET in the image,
    the way the operator does by eye. Yaw the wrist to the measured base (fold-aware, base x CAD seat vs held part),
    hover 0.10 m over the predicted seat, then marker_yaw.socket_offset: the part's real top-face edges vs its CAD top
    face at the seat raised to the current height, placed by the base's own markers in the same frame (no hand-eye,
    no held-pose chain); move by minus it, repeat (<= 4) until < 1 mm. Returns (base pose to aim at = shifted by the
    total move, held quat after the yaw turn). SERVO_MEASURED[obj] says whether a reading was taken (else only the bias)."""
    SERVO_MEASURED[obj] = False
    if observed_base is None or held_q is None:
        return observed_base, held_q
    from compliant_insertion_studio.wrapper.marker_yaw import socket_shift_world, held_outline_px, still_frame
    from compliant_insertion_studio.wrapper.align_at_hover import rel_yaw_error, turn_wrist
    from compliant_insertion_studio.wrapper.cad_lookup import predict_tcp_at_seat
    from primitives.shared.config import held_tool_offset_for
    (bx, by, bz), bq = observed_base
    err, _t = rel_yaw_error(obj, base, R_.from_quat(bq), R_.from_quat(held_q))
    prior = servo_prior(obj) if SERVO_PRIOR else None
    prior_yaw = prior[1] if prior is not None else 0.0
    if abs(err + prior_yaw) > 0.3 and turn_wrist(err + prior_yaw):
        held_q = list((R_.from_euler('z', err + prior_yaw, degrees=True) * R_.from_quat(held_q)).as_quat())
        say(f"  [servo] {obj}: wrist turned {err + prior_yaw:+.2f} deg to the base yaw"
            + (f" ({prior_yaw:+.2f} of it the servo prior)" if prior is not None else ""))
    elif prior is not None:
        prior_yaw = 0.0                                     # no turn taken: none of the prior's yaw applied
    # the base read again WITH the turned wrist: the servo projects it from the hover with this same orientation, so the
    # hand-eye's translation error cancels between the read and the projection
    observed_base = track_base_in_hand(base, observed_base, say, log_fh, held_obj=obj)
    LAST_TRACK_READ['pose'] = observed_base
    (bx, by, bz), bq = observed_base
    tcp = (snapshot_robot_state().get('tcp') or {})
    pred = predict_tcp_at_seat(base_name=base, object_name=obj, grasp_id=grasp_id, base_world_xyz=[bx, by, bz],
                               base_world_quat_xyzw=bq, held_quat_xyzw=held_q, ee_orientation_xyzw=tcp['quat_xyzw'],
                               flange_offset_m=float(held_tool_offset_for(obj)[2]))
    st = np.array(pred['predicted_tcp_at_seat']['xyz_m']); q = tcp['quat_xyzw']
    # the held part's place in the image, once per grasp, over bare table beside the base (camera rigid with the gripper)
    _emitter_hold()                                     # colour-only from the outline on (the base was read before it)
    r = _move_unless_there((bx + 0.17, by, 0.32), list(q), 'servo: outline view over table', log_fh)   # in process
    _emitter(False)
    try:
        img_, _ = still_frame(n_avg=4)
    finally:
        _emitter(True)
    outline = held_outline_px(img_) if (img_ is not None and r.get('result') != 'failure') else None
    if outline is None:
        say(f"  [servo] {obj}: held part not outlined over the table - aiming as read"); return observed_base, held_q
    say(f"  [servo] {obj}: held part outlined ({len(outline['far'])} far / {len(outline['left'])} left edge points)")
    _emitter_hold()                                                   # colour-only from here: no depth read follows
    # evidence for every servo: the outline, the table frame and each round's overlay (target green, outline magenta)
    sv_stem = REPO_ROOT / 'compliant_insertion_studio' / 'logs' / f"servo_{obj}_{time.strftime('%Y%m%d_%H%M%S')}"
    try:
        import cv2 as _cv2
        json.dump(outline, open(f"{sv_stem}_outline.json", 'w')); _cv2.imwrite(f"{sv_stem}_table.png", img_)
    except Exception:
        pass
    # measure LOW, 2.5 mm over the rim: the target (part at its seat raised by the lift) is projected from the base
    # pose read in the SAME frame, so what remains is the hand-eye's tilt error x the lift (at 0.10 m: 1 deg -> 1.7 mm;
    # at 0.035 m: 0.6 mm). Lateral moves and wrist turns happen 12 mm over the rim.
    try:
        # the FIRST surface the held part can meet: the base top or the top of any part seated before it (10-22 01:59:
        # inverted_u_yellow nests over the seated parts; measuring 2.5 mm over the base top hit them, 4 N guard)
        from compliant_insertion_studio.wrapper.cad_geometry import build_contact_candidates, get_seated_so_far
        # only parts seated UNDER the held part's footprint can meet it (10-22 02:36: u_orange's footprint does not
        # overlap u_brown's, yet u_brown's top set the measuring height 70 mm up and the base markers left the view)
        under = [o for o in get_seated_so_far(base, obj) if _footprints_overlap(base, obj, o)]
        cands = build_contact_candidates(target_object=obj, target_grasp_id=int(grasp_id), base_name=base,
                                         base_world_z=float(bz), R_grasp_quat_xyzw=held_q,
                                         predicted_tcp_z_at_seat=float(st[2]),
                                         seated_objects=under,
                                         flange_offset_world_z=-float(held_tool_offset_for(obj)[2]))
        rim_z = max(v for k, v in cands.items() if k != 'SEAT')
        say(f"  [servo] {obj}: first contact {1000*(rim_z - st[2]):+.1f} mm over the seat "
            f"({', '.join(f'{k} {1000*(v - st[2]):+.1f}' for k, v in cands.items() if k != 'SEAT')})")
    except Exception as e:
        rim_z = float(st[2]) + 0.031
        say(f"  [servo] {obj}: no CAD contact candidates ({e}) - first contact assumed 31 mm over the seat")
    z_meas = rim_z + SERVO_MEAS_OVER_RIM_M
    # lateral moves and wrist turns happen at the old hover height (100 mm over the seat): parts already seated stand up
    # to ~40 mm over the base top, and a held bar reaches far beyond the fingers (10-22 01:18: a diagonal move from the
    # table view down to 12 mm over the rim drove line_green into the seated U parts - protective stop). Only the last
    # leg to the measuring height is vertical, at 4 N guard.
    z_clear = max(float(st[2]) + HOVER_ALIGN_ABOVE_M, rim_z + SERVO_CLEAR_OVER_RIM_M)
    from compliant_insertion_studio.wrapper.marker_yaw import base_pose_same_frame
    from seams import servo_law as SL
    xy0 = None
    if prior is not None:
        pw = R_.from_euler('z', float(R_.from_quat(bq).as_euler('xyz')[2])).apply([prior[0][0], prior[0][1], 0.0])[:2]
        xy0 = st[:2] + pw
        say(f"  [servo] {obj}: starting at the servo prior ({1000*pw[0]:+.1f}, {1000*pw[1]:+.1f}) mm, {prior_yaw:+.2f} deg "
            f"(median of its last {prior[2]} servos)")
    law = SL.HoverServoLaw(st[:2], xy=xy0)      # the decisions (AR3 step 1); this function moves and reads
    if prior_yaw:
        law.turned(prior_yaw)                       # the prior's turn is part of the image yaw the insert's base takes
    xy = law.xy
    for k in range(SL.SERVO_ITERS + 1):
        cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
        if cur and cur[2] < z_clear - 0.002:                       # straight up first, never sideways at low height
            _move_unless_there((cur[0], cur[1], z_clear), q, f'servo up {k}', log_fh)
            if _SAFETY_STOP:
                return observed_base, held_q
        r = _move_unless_there((xy[0], xy[1], z_clear), q, f'servo clear {k}', log_fh)
        if r.get('result') != 'failure':
            r = _move_unless_there((xy[0], xy[1], z_meas), q, f'servo hover {k}', log_fh, extra=('--max-force', '4', '--min-duration', '2'))
        if r.get('result') == 'failure' and r.get('contact') and not _SAFETY_STOP:
            # touched down before the measuring height (10-22 12:05, u_orange twice at ~4.5 N: the grasp left it hanging
            # lower than the chain) - the image comparison projects the target at whatever height we are, so measure
            # from 2 mm over the contact instead of giving up (attempt 17 measured 4.5 mm and corrected it; 18 did not)
            cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
            if cur:
                z_meas = float(cur[2]) + 0.002
                r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                    f"{xy[0]:.5f}", f"{xy[1]:.5f}", f"{z_meas:.5f}", '--quat', *[f"{v:.6f}" for v in q],
                                    '--min-z', '0.15'], label=f'servo off the contact {k}', timeout_s=60, log_fh=log_fh)
                law.measuring_low()
                say(f"  [servo] {obj}: touched down {1000*(z_meas - 0.002 - rim_z):+.1f} mm over the CAD rim - measuring "
                    f"from 2 mm above the contact")
                # the touchdown MEASURES the grasp depth: the part hangs that much lower in the jaws than the chain says.
                # Carry it everywhere the depth matters - the seat and rim heights here (the image projects the target
                # at the part's real lift), and the insert (drive_align reports it; drive-mate inserts with it).
                # Attempt 26 (10-22 18:01): u_orange touched down +7.4 mm; the servo projected at the chain's lift and
                # oscillated (+10.6 -> +1.1 -> -3.0 -> +5.1 mm), the insert's 61 probes all met the rim +7.2 mm high
                # and its rim-measured depth fix (capped at 6 mm) ignored it.
                dz_ = (z_meas - 0.002) - rim_z
                if GRASP_DEPTH_MEAS_MIN_M < dz_ <= GRASP_DEPTH_MEAS_MAX_M and obj not in GRASP_DEPTH_FIX:
                    from primitives.shared.config import held_tool_offset_for
                    d0 = float(held_tool_offset_for(obj)[2])
                    os.environ['GRASP_TOOL_OFFSET_M'] = f"{d0 + dz_:.5f}"
                    GRASP_DEPTH_FIX[obj] = dz_
                    st = st + np.array([0.0, 0.0, dz_]); rim_z += dz_
                    z_clear = max(float(st[2]) + HOVER_ALIGN_ABOVE_M, rim_z + SERVO_CLEAR_OVER_RIM_M)
                    say(f"  [servo] {obj}: grasp depth {1000*d0:.1f} -> {1000*(d0 + dz_):.1f} mm (measured by the touchdown); "
                        f"seat and rim raised {1000*dz_:+.1f} mm")
        if r.get('result') == 'failure':
            say(f"  [servo] {obj}: hover not reached ({r.get('error', '')})")
            if _SAFETY_STOP:
                return observed_base, held_q                            # nothing more is sent on a stopped robot
            break
        _emitter(False)
        try:
            img_, _ = still_frame(n_avg=4); tcp_now = (snapshot_robot_state().get('tcp') or {})
            tp = (np.asarray(tcp_now['xyz']), np.asarray(tcp_now['quat_xyzw']))
            base_sf, sfinfo = base_pose_same_frame(base, img_, tp, bq) if img_ is not None else (None, {'error': 'no frame'})
            lift = float(tp[0][2] - st[2])
            d, info = socket_shift_world(obj, base, base_sf if base_sf is not None else observed_base, lift, outline, img=img_,
                                         tcp_pose=tp, debug_path=f"{sv_stem}_r{k}.png")
        finally:
            _emitter(True)
        if base_sf is None and z_meas < z_clear - 0.01:
            # the seat lies on the far side of the base for this view: from just over the rim the base fills the frame and
            # its corner markers leave it (10-22 02:52: u_orange, 0 markers); from the clear height 2-3 are in view.
            # Measure from there instead (the hand-eye tilt lever grows to ~1.7 mm/deg; the dip grid covers +-1.5 mm)
            say(f"  [servo] {obj} round {k}: {sfinfo.get('error')} over the rim - measuring from the clear height instead")
            z_meas = z_clear
            law.measuring_from_clear(float(R_.from_quat(bq).as_euler('xyz', degrees=True)[2]))   # the 2x2 response, at this base yaw
            r = _move_unless_there((xy[0], xy[1], z_clear), q, f'servo up to measure {k}', log_fh)
            if _SAFETY_STOP:
                return observed_base, held_q
            _emitter(False)
            try:
                img_, _ = still_frame(n_avg=4); tcp_now = (snapshot_robot_state().get('tcp') or {})
                tp = (np.asarray(tcp_now['xyz']), np.asarray(tcp_now['quat_xyzw']))
                base_sf, sfinfo = base_pose_same_frame(base, img_, tp, bq) if img_ is not None else (None, {'error': 'no frame'})
                lift = float(tp[0][2] - st[2])
                d, info = socket_shift_world(obj, base, base_sf if base_sf is not None else observed_base, lift, outline,
                                             img=img_, tcp_pose=tp, debug_path=f"{sv_stem}_r{k}_clear.png")
            finally:
                _emitter(True)
        if base_sf is None:
            # fail closed (10-22 02:25: with 0-1 base markers in view the fallback projected the base read - the biased
            # path - and its 27-70 px-spread fits turned u_orange's wrist 6 x -5 deg = -30 deg into the insert)
            say(f"  [servo] {obj} round {k}: base not read in the frame ({sfinfo.get('error')}) - no correction, aiming as read")
            break
        act = law.reading(d, info)
        for n_ in law.notes:
            say(f"  [servo] {obj} {n_}")
        law.notes.clear()
        if isinstance(act, SL.Stop) and act.why.startswith('not measured'):
            say(f"  [servo] {obj} round {k}: not measured ({info.get('error')})"); break
        if isinstance(act, SL.Stop) and act.why.startswith('fit spread'):
            say(f"  [servo] {obj} round {k}: fit spread {info.get('spread_px')} px > {SL.SERVO_MAX_SPREAD_PX} - the outline "
                f"does not match the target, no correction"); break
        say(f"  [servo] {obj} round {k}: part {info['dxy_mm']} mm from over its socket, yaw {info.get('yaw_img_deg')} deg in the image "
            f"(shift {info['shift_px']} px, spread {info['spread_px']} px; base in frame {json.dumps(sfinfo)})")
        if isinstance(act, SL.Converged):
            break
        if isinstance(act, SL.Stop):
            say(f"  [servo] {obj}: {act.why} - more than the reads can be off, stopping the servo"); break
        if isinstance(act, SL.Turn):
            _move_unless_there((xy[0], xy[1], z_clear), q, f'servo up for turn {k}', log_fh)   # vertical: xy is where we measured
            if _SAFETY_STOP:
                return observed_base, held_q
            if turn_wrist(act.deg):
                law.turned(act.deg)
                if act.xy is not None:
                    xy = np.array(act.xy)                           # the step taken with the turn
                held_q = list((R_.from_euler('z', act.deg, degrees=True) * R_.from_quat(held_q)).as_quat())
                q = (snapshot_robot_state().get('tcp') or {}).get('quat_xyzw') or q
                say(f"  [servo] {obj}: wrist turned {act.deg:+.2f} deg (image yaw)")
            else:
                law.xy = np.array(xy, float)                        # no turn, so no step either: re-measure here
            law.next_round()
            continue                                                # re-measure xy after the turn
        xy = np.array(act.xy)                                       # SL.Step: the next place to measure
        law.next_round()
    out = law.finish()
    xy = law.xy
    if isinstance(out, SL.Aim) and out.overshot:
        say(f"  [servo] {obj}: last step overshot - aiming from the best reading ({1000*out.overshot[0]:+.1f}, "
            f"{1000*out.overshot[1]:+.1f}) mm at gain {out.overshot[2]:.2f}")
    _move_unless_there((xy[0], xy[1], z_clear), q, 'servo done: clear', log_fh)
    SERVO_MEASURED[obj] = law.final_reading is not None
    if isinstance(out, SL.Stop):
        say(f"  [servo] {obj}: {out.why} - not applied")
        return observed_base, held_q
    c = np.array(out.dxy); yaw_total = out.yaw_deg
    if law.final_reading is not None:
        _record_servo(obj, c, yaw_total, bq, law.k)
    if yaw_total:
        # the image yaw becomes the base's yaw for the insert, turned about the SEAT point (the pocket stays put): the
        # insert's own yaw step (held part vs base x seat) then has nothing to undo
        from compliant_insertion_studio.wrapper.cad_lookup import load_seat_pose_in_base
        sp_, _ = load_seat_pose_in_base(base, obj)
        Rb0 = R_.from_quat(bq); sw = np.array([bx, by, bz]) + Rb0.apply(sp_)
        Rb1 = R_.from_euler('z', yaw_total, degrees=True) * Rb0
        bx, by, bz = sw - Rb1.apply(sp_); bq = tuple(Rb1.as_quat())
        say(f"  [servo] {obj}: base yaw for the insert turned {yaw_total:+.2f} deg about the seat (image)")
    say(f"  [servo] {obj}: aim moved by ({1000*c[0]:+.1f}, {1000*c[1]:+.1f}) mm")
    return ((bx + c[0], by + c[1], bz), bq), held_q


from seams.servo_law import (SERVO_ITERS, SERVO_YAW_TOL_DEG, SERVO_YAW_MIN_TURN_DEG, SERVO_MAX_SPREAD_PX,  # noqa: E402,F401
                             SERVO_MAX_YAW_DEG, SERVO_GAIN_CLEAR, SERVO_YAW_GAIN, SERVO_BIAS_M, SERVO_BIAS_CLEAR_M,
                             SERVO_TOL_M, SERVO_MAX_M)   # one definition (the law's); depth_servo uses them too
SERVO_MEAS_OVER_RIM_M = 0.004  # measuring height over the CAD rim contact (the real rim reads ~1.5 mm higher: ~2.5 mm clear)
SERVO_CLEAR_OVER_RIM_M = 0.012 # lateral moves and wrist turns happen here


_EMITTER = {'state': None, 'hold': False}


def _emitter(on):
    """The D415 projector's dots swamp the colour image in a dark room (10-21): off for colour-only reads. Each set is a
    ros2 CLI call + 0.8 s settle (~2.2 s): it is skipped when the projector is already so, and held off through a servo's
    colour-only rounds (_emitter_hold) - run 41's u_brown align toggled it 8 times (~18 s)."""
    if on and _EMITTER['hold']:
        return
    if _EMITTER['state'] is not None and _EMITTER['state'] == bool(on):
        return
    try:
        if not (EMITTER_INPROC and _set_emitter_inproc(on)):
            subprocess.run(['ros2', 'param', 'set', '/camera/camera', 'depth_module.emitter_enabled', '1' if on else '0'],
                           capture_output=True, timeout=10, env=make_env())
        time.sleep(0.8)
        _EMITTER['state'] = bool(on)
    except Exception:
        _EMITTER['state'] = None


EMITTER_INPROC = os.environ.get('EMITTER_INPROC', '1') == '1'


def _set_emitter_inproc(on):
    """The projector set as one SetParameters service call on a private context (~0.05 s) - the ros2 CLI took ~1.5 s
    to start each time (10-23). False when the service did not answer: the caller falls back to the CLI."""
    import rclpy
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from rclpy.executors import SingleThreadedExecutor
    ctx = rclpy.context.Context()
    rclpy.init(context=ctx)
    try:
        n = rclpy.create_node('replay_emitter', context=ctx)
        ex = SingleThreadedExecutor(context=ctx); ex.add_node(n)
        cl = n.create_client(SetParameters, '/camera/camera/set_parameters')
        if not cl.wait_for_service(timeout_sec=2.0):
            return False
        req = SetParameters.Request()
        req.parameters = [Parameter(name='depth_module.emitter_enabled',
                                    value=ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=1 if on else 0))]
        fut = cl.call_async(req)
        ex.spin_until_future_complete(fut, timeout_sec=3.0)
        res = fut.result()
        return bool(res and res.results and res.results[0].successful)
    finally:
        try:
            n.destroy_node()
        except Exception:
            pass
        rclpy.shutdown(context=ctx)


def _emitter_hold():
    """Projector off until _emitter_release: the frames from here on are colour-only (no depth read in between)."""
    _EMITTER['hold'] = False
    _emitter(False)
    _EMITTER['hold'] = True


def _emitter_release():
    _EMITTER['hold'] = False
    _emitter(True)


def certify_once_at_end(objs, base, observed_base, say, log_fh=None):
    """ExecuteMate for every mate in assembly order from ONE top-down view over the base (no motion between calls,
    nothing moved on a refusal). Returns {obj: execute_mate output}."""
    from seams.client import mate_for
    from base_depth import VIEW_FLANGE_Z as _VZ
    (bx, by, _), _ = observed_base
    out = {}
    for q in (('0', '1', '0', '0'), ('1', '0', '0', '0')):    # top-down, or pi-turned if wrist_3 would wind past 4 rad
        r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz', f"{bx:.4f}",
                            f"{by:.4f}", f"{_VZ:.2f}", '--quat', *q], label='certify view (one, over the base)',
                           timeout_s=60, log_fh=log_fh)
        if r.get('result') != 'failure':
            break
    else:
        say("  [certify] the view over the base was not reached - nothing certified"); return out
    for obj in objs:
        mate = mate_for(obj, base)
        if mate is None:
            continue
        o = _seam_call(['execute', '--mate', mate], 450)
        out[obj] = o
        if o.get('placed'):
            SEATED.mark(obj, True, how='execute_mate',
                        depth_mm=None if o.get('engagement_depth_m') is None else 1000 * float(o['engagement_depth_m']))
        say(f"  [certify] {mate} ({obj}): placed={o.get('placed')} refusal={o.get('refusal')!r} "
            f"{o.get('refusal_detail', '')} depth={o.get('engagement_depth_m')}")
    return out


def certify_via_execute_mate(obj, base, observed_base, say, log_fh=None):
    """SEAMS_CERTIFY=1 (M3a): after a part is done, fly the seat view and ask /ur5e/execute_mate. Log-only until FMB1
    proves it: the replay's own verification still decides."""
    from seams.client import mate_for
    import seat_check as _sc
    mate = mate_for(obj, base)
    if mate is None or observed_base is None:
        return None
    out = None
    # the view that measured this mate last time first (runs 32b/33b: the yellow's outer view read an end below the floor
    # both times, ~25 s each) - learned from results, not a per-part table
    vfile = REPO_ROOT / 'ablations' / 'logs' / 'certify_views.json'
    try:
        vd_ = json.load(open(vfile))
    except Exception:
        vd_ = {}
    # ranked by each view's record for this mate (measured / not), the last one that measured breaking ties: the last view
    # alone alternated u_orange between 'inner' and 'along+' (each reads an end 2 mm under the floor at some base poses)
    # and tried 'outer', which it never reaches, every time one of them failed (runs 63, 65: ~13 s a wasted view)
    ent = vd_.get(mate)
    last_side = ent if isinstance(ent, str) else (ent or {}).get('last')
    stats = (vd_.get('_stats') or {}).get(mate) or {}

    def _rank(v):
        st = stats.get(v) or {}
        return (-(st.get('ok', 0) + 1) / (st.get('ok', 0) + st.get('fail', 0) + 2), 0 if v == last_side else 1)
    sides = sorted(_sc.VIEW_SIDES, key=_rank)

    def _note(side, ok):
        try:
            d_ = json.load(open(vfile)) if vfile.exists() else {}
            st_ = d_.setdefault('_stats', {}).setdefault(mate, {}).setdefault(side, {'ok': 0, 'fail': 0})
            st_['ok' if ok else 'fail'] += 1
            if ok:
                d_[mate] = side
            json.dump(d_, open(vfile, 'w'), indent=1)
        except Exception:
            pass
    for side in sides:                              # vision does no motion: the planner tries views until it measures
        r = run_subprocess(_sc.view_pose(obj, observed_base, base, side), label=f'execute_mate: seat view ({obj}, {side})',
                           timeout_s=90, log_fh=log_fh)
        if r.get('result') == 'failure':
            _note(side, False)
            say(f"  [execute_mate] {mate}: the {side} view was not reached - next view")
            continue
        out = _seam_call(['execute', '--mate', mate], 450)
        if out.get('refusal') != 'engagement_invalid':
            if out.get('engagement_depth_m') is not None:
                _note(side, True)
            break
        _note(side, False)
        say(f"  [execute_mate] {mate}: not measurable from the {side} view ({out.get('refusal_detail', '')}) - next view")
    if out is None:
        say(f"  [execute_mate] {mate}: no seat view reached - not asked"); return None
    if out.get('placed'):
        SEATED.mark(obj, True, how='execute_mate',
                    depth_mm=None if out.get('engagement_depth_m') is None else 1000 * float(out['engagement_depth_m']))
    say(f"  [execute_mate] {mate}: placed={out.get('placed')} refusal={out.get('refusal')!r} "
        f"{out.get('refusal_detail', '')} depth={out.get('engagement_depth_m')}")
    return out


DEPTH_VIEW_ZS = (0.52, 0.45)         # flange heights for the pre-pick depth read (the D415 returns no depth < ~0.45 m)
DEPTH_PICK_MAX_MOVE_M = 0.005        # 10-21 17:19: 10 -> 5 mm. Good fits today moved <= 3.8 mm from the marker seed (12
                                     # picks); u_brown lying on its side slid 8.5 mm along its uniform top face, was held
                                     # 8.5 mm off and the 8 mm spiral missed its pocket twice (rim, +34 mm).
                                     # Earlier: 10 mm - a fit this far from its marker seed is not trusted (in clutter - a bar in the base -
DEPTH_PICK_MAX_TURN_DEG = 3.0       # 10-21 17:25: good fits turned <= 2.3 deg; a side-lying u_brown's turned 10.0 (seed 179.9)
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
    cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
    if cur and np.hypot(cur[0] - tr[0], cur[1] - tr[1]) < 0.005 and abs(cur[2] - DEPTH_VIEW_ZS[0]) < 0.005:
        ok = True             # the part view just put the camera here (same pose: a 3.7 s move + settle saved)
    for vz in ([] if ok else DEPTH_VIEW_ZS):
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
        if pose is None and 'no marker read' in str(info.get('error', '')):
            # dark room (10-22 01:46: brightness 31): markers need the 0.40 m view, depth needs >= 0.45 m. Seed from a
            # marker read at 0.40 m straight below, then fit the depth from the depth height with that seed.
            cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
            seed = None
            if cur:
                r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                    f"{cur[0]:.4f}", f"{cur[1]:.4f}", f"{PART_VIEW_Z_LOW_LIGHT:.4f}", '--quat', '0.0', '1.0',
                                    '0.0', '0.0', '--min-z', '0.15'], label=f'depth-pick seed view above {obj} (low light)',
                                   timeout_s=60, log_fh=log_fh)
                S_ = (observe_many([obj], 8.0, n_samples=8).get(obj) or []) if r.get('result') != 'failure' else []
                if S_:
                    seed = (tuple(np.median(np.array([s_['xyz'] for s_ in S_]), axis=0)), tuple(S_[-1]['quat_xyzw']))
                    run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                    f"{cur[0]:.4f}", f"{cur[1]:.4f}", f"{cur[2]:.4f}", '--quat', '0.0', '1.0', '0.0', '0.0',
                                    '--min-z', '0.15'], label=f'depth-pick view above {obj} (back up for depth)',
                                   timeout_s=60, log_fh=log_fh)
                    pose, info = _read(obj, seed=seed)
                    info = dict(info, seed='marker read at 0.40 m (low light)')
    except Exception as e:
        pose, info = None, {'error': str(e)}
    say(f"  [depth-pick] {obj}: {json.dumps(info)}")

    def trusted(pose_, info_):
        return pose_ is not None and not (
            np.hypot(*info_.get('moved_mm', [0, 0])) > 1000 * DEPTH_PICK_MAX_MOVE_M
            or abs(info_.get('turned_deg', 0.0)) > DEPTH_PICK_MAX_TURN_DEG
            or info_.get('n', 0) < DEPTH_PICK_MIN_MATCHED * max(info_.get('n_data', 1), 1))
    if not trusted(pose, info):
        # re-measure once from the other view height (a different depth sample of the same part) before refusing
        cur = (snapshot_robot_state().get('tcp') or {}).get('xyz')
        alt = next((z_ for z_ in DEPTH_VIEW_ZS if cur is None or abs(z_ - cur[2]) > 0.02), None)
        if cur is not None and alt is not None:
            r = run_subprocess(['python3', '-u', '-m', 'primitives.move_cartesian', '--mode', 'real', '--xyz',
                                f"{cur[0]:.4f}", f"{cur[1]:.4f}", f"{alt:.4f}", '--quat', '0.0', '1.0', '0.0', '0.0',
                                '--min-z', '0.15'], label=f'depth-pick re-measure above {obj} at {alt:.2f} m', timeout_s=60,
                               log_fh=log_fh)
            if r.get('result') != 'failure':
                try:
                    from part_depth import read as _read
                    pose, info = _read(obj)
                except Exception as e:
                    pose, info = None, {'error': str(e)}
                say(f"  [depth-pick] {obj} (re-measured at {alt:.2f} m): {json.dumps(info)}")
    if not trusted(pose, info):
        say(f"  [depth-pick] {obj}: fit {'failed' if pose is None else 'moved ' + str(info.get('moved_mm')) + ' mm, turned ' + str(info.get('turned_deg')) + ' deg'}"
            f" - not trusted (no marker fallback)"); return None
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
            state.aim = None
            argv += ['--insert', '--use-default-base-position',
                     '--insertion-type', 'compliant']
            # grasp_id is required by the compliant_insert path. Chain from the
            # most recent move_to_grasp; fall back to 1 if not seen.
            gid = state.last_grasp_id if state.last_grasp_id is not None else 1
            argv += ['--grasp-id', str(gid)]
            argv += state.quat_argv()
            if observed_base is not None:
                # The aim (seams.aim, AR1): the measured base + follow / seat-history bias / yaw experiment; the
                # per-part seat z rides in argv only (translate_object's convention). drive-mate takes state.aim.
                obj_ = kw['object_name']
                aim = aim_for_insert(obj_, RUN_BASE['name'], observed_base, load_facts())
                for line in (*aim.notes, *(c.detail for c in aim.corrections)):
                    print(line, flush=True)
                if aim.shift('bias') != (0.0, 0.0):
                    _APPLIED_BIAS[obj_] = aim.shift('bias')
                state.aim = aim
                px, py, pz = aim.argv_pos()
                argv += ['--final-base-pos', f"{px:.6f}", f"{py:.6f}", f"{pz:.6f}",
                         '--final-base-orientation', *[f"{v:.6f}" for v in aim.quat]]
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
CROSS_LOW_EDGE_MAX_MM = 3.0               # ... and its other edge still near the seat (a lip carries one edge only)


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


def _wedged_near_seat(obj, since, result=None):
    """{'dz_mm', 'dxy_mm'} if the last insert ended wedged (still, 0 < dz <= 7 mm over its predicted seat, within 5 mm
    in xy: on a channel edge), else None. The insert's own outcome when it returned one (drive-mate: judged at rest),
    else its trace judged by seams.insert_outcome (both meta schemas; the old reader saw only predicted_seat_xyz)."""
    from seams.insert_outcome import outcome_of
    if result is not None and result.get('stage_reached'):
        o = dict(result)
    else:
        o = outcome_of('', obj, since).as_dict()
    if o.get('stage_reached') != 'wedged' or o.get('measured_correction') is None:
        return None
    d = o['measured_correction']
    return {'dz_mm': 1000 * float(d[2]), 'dxy_mm': 1000 * float(np.hypot(d[0], d[1]))}


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
    if min(lo_edge, hi_edge) > CROSS_LOW_EDGE_MAX_MM:
        # both edges up: the whole end rests on the base top past its pocket - the axial push (10-14 line_green fix),
        # not a lip. 10-22 09:35: edges +7.1/+8.9 called a lip, the sideways nudge lowered onto the seated U part
        # beside line_green twice (7.9 / 7.7 N, stopped)
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
NUDGE_FIRST_END_MM = 4.0             # depth end this high after an insert: a wedge - the nudge before any tap
NUDGE_SEAT_OVER_MM = 2.0             # ...and never past the seated CAD end by more than this: line_green has no axial
                                     # stop - run 31's push met no force, ran the 15 mm cap and left the bar 18 mm past its
                                     # seat, against the U at its far end (u_orange then read outside its seat)
NUDGE_TRIES = 3
NUDGE_SHIFT_MM = 3.0                 # axial shift that means an end rides past its pocket
NUDGE_END_DIFF_MM = 2.5              # one end this much higher than the other: nudge, not reseat
NUDGE_START_GAPS_MM = (10.0, 20.0, 30.0)   # finger block this far past the depth end; next on contact
NUDGE_PAD_Z_BASE_M = 0.037           # pad centre height (base frame): tips 22 mm, base top 17.5 mm
NUDGE_AXIAL_BIAS_MM = 7.0          # depth puts a raised end ~7 mm past the real one (edge smear on
                                   # the D415). Not a base-read error: with the depth base and 0 here,
                                   # step 1 missed the bar entirely (operator, 2026-10-15)


LAST_SEAT_RES = {}             # obj -> the last depth seat check (the ladder's final word when it runs out)


def _seat_check_accepted(obj, ob, **kw):
    """seat_check.check, passing only what ExecuteMate would certify, one read-sigma inside its band (seams.acceptance:
    one rule for both - attempt 24 stopped its recovery at an end +2.0 mm the certifier then refused)."""
    from seat_check import check as _check
    res = _check(obj, ob, **kw)
    LAST_SEAT_RES[obj] = res
    if res.get('ok') and res.get('height_err_mm') is not None:
        from seams.acceptance import margin_m, seat_inside
        inside, why = seat_inside(kw.get('base') or RUN_BASE['name'], obj, float(res['height_err_mm']) / 1000.0,
                                  res.get('end_heights_mm') or [], margin=margin_m())
        if not inside:
            res = dict(res, ok=False, acceptance=why)
            print(f"  [seat-check] {obj}: seat check passes it, the certifier would not ({why}) -> recovery goes on",
                  flush=True)
    return res


def nudge_into_slot(obj, ob, res, say, out_dir) -> bool:
    """Push a tilted bar's raised end along its axis into its slot (operator's method): closed
    fingers lowered beside the high end from safe height, then ONE guarded push toward the slot
    that stops at NUDGE_MAX_FORCE_N — the bar slides while it is free and the force rises once it
    is blocked (seated against the pocket end, or stuck). Depth re-check after every try; the next
    try starts from the new measurement. The base is re-fitted by every check (7th-axis creep)."""
    if not BASE_FIXED_CFG() and os.environ.get('NUDGE_FREE_BASE', '0') != '1':
        # on a free base the push never blocks - the base slides first: attempt 27's three pushes ran their travel cap
        # and moved base1 ~50 mm while the bar was seated (tap map). Vertical presses (seams.tap_probe) only.
        say(f"  [nudge] {obj}: free base - no lateral push (NUDGE_FREE_BASE=1 to allow)")
        return False
    check = _seat_check_accepted
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
        face_mm = end_mm - sgn * NUDGE_PUSH_PAST_MM
        if f.get('half_long_m'):                                     # the seated end bounds it (no axial stop to meet)
            face_mm = sgn * max(sgn * face_mm, 1000 * float(f['half_long_m']) - NUDGE_SEAT_OVER_MM)
        push = world((face_mm + sgn * half) / 1000.0)
        say(f"  [nudge] try {k}: {'+y' if sgn > 0 else '-y'} end high, depth end {raw_mm:+.0f} mm (base) "
            f"-> push toward the slot until {NUDGE_MAX_FORCE_N:g} N (finger face to {face_mm:+.0f}: at most "
            f"{NUDGE_PUSH_PAST_MM:g} mm past {end_mm:+.0f}, {NUDGE_SEAT_OVER_MM:g} mm past the seated end)")
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
            _record_seat_check_error(obj, res)     # attempt 23: only the first-pass and after-taps passes recorded it
            return True
        if _level_but_high(res) or abs(res.get('roll_deg', 0.0)) >= ROLL_PRESS_DEG:
            say(f"  [nudge] {obj}: level along it now but high/rolled - stop nudging, edge press next")
            return False
    return False


EDGE_OFFSET_MM = 8.0                 # tap/press point off the bar's centreline (26 mm bar, 20 mm block)
EDGE_TAP_N = 3.0                     # measuring tap: a 3 N centre tap did not move the wedged bar
EDGE_PRESS_N = 4.0                   # guard under the operator's 5 N limit: presses overshoot by up to 0.7 N (run 32)
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
    check = _seat_check_accepted
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
    if res.get('ok'):
        _record_seat_check_error(obj, res)
    return bool(res.get('ok'))


def _record_seat_check_error(obj, res):
    """A passed depth seat check measures where the part really sits ACROSS its slot (cross centre) and along it (axial
    shift), base frame, mm: written to the verify ledger as rel_err_mm so the part threading over it next is aimed at it
    (_follow_seated_general). 13:39/13:54 the yellow, aimed at its CAD seat, came down on line_green's top twice -
    line_green sat 1.7 mm across from CAD (its check), the yellow straddles it with +-1 mm."""
    try:
        f = res.get('face') or {}
        cross = float((res.get('cross') or {}).get('centre_mm'))
        along = float((res.get('axial') or {}).get('shift_mm', 0.0))
        e = [cross, along] if f.get('long_axis', 'y') == 'y' else [along, cross]
        SEATED.set_seat_error(obj, e, 'depth seat check', touch_t=True)
        print(f"  [seat-check] {obj}: measured seat error {e} mm (base frame) recorded for the parts over it", flush=True)
    except Exception as ex:
        print(f"  [seat-check] {obj}: seat error not recorded ({ex})", flush=True)


def _seat_by_taps(obj, args, say):
    """The seat judged by taps (operator 10-22: "use force to tap smartly - you know where it seats and the force
    feedback"): one depth view - it re-fits the base and measures where the part sits ACROSS its slot (what the parts
    over it follow) - then seams.tap_probe maps the top against the seated CAD and presses only what is high, <= 4.5 N,
    vertical. Attempt 27: the depth check called line_green's end +1.5/+1.7/+1.9 mm high, three lateral nudges pushed
    the free base ~50 mm, and the taps then read the bar seated (-1.35..+0.05 mm). True / False; None: the taps could
    not judge it (taps off the part) - the depth ladder decides."""
    import seat_check as _sc
    from seams.tap_probe import tap_recover
    out_dir = REPO_ROOT / 'ablations' / 'logs' / 'seat_checks'
    res = _sc.check(obj, args.observed_base, move=True, out_dir=out_dir, base=RUN_BASE['name'])
    ob = res.get('observed_base') or args.observed_base
    args.observed_base = ob
    # the taps go where drive-mate last MEASURED the base (its align read, top-down), not the depth re-fit from the
    # seat check's tilted view: run 28c's taps from that re-fit all landed beside the bar (no contact at any point)
    ob = getattr(args, 'measured_base', None) or ob
    say(f"  [seat-check] {obj} (depth, for the base and the across error): "
        f"{json.dumps({k: v for k, v in res.items() if k not in ('profile_mm', 'observed_base')})}")
    ends_ = res.get('end_heights_mm') or []
    if res.get('height_err_mm') is not None:
        from seams.acceptance import seat_inside
        inside_, why_ = seat_inside(RUN_BASE['name'], obj, float(res['height_err_mm']) / 1000.0, ends_)
        if inside_:
            # depth already passes it by the certifier's own rule: the taps would only confirm a seat (run 38b: line_green
            # seated from the insert, ends +1.5 / -0.2 mm, 49 s of taps) - ExecuteMate measures it next anyway
            say(f"  [seat-check] {obj}: depth passes it ({why_}) - no taps")
            if (res.get('cross') or {}).get('centre_mm') is not None:
                _record_seat_check_error(obj, res)
            SEATED.mark(obj, True, how='depth seat check')
            return True
    from compliant_insertion_studio.wrapper.cad_geometry import wedge_possible
    if wedge_possible(RUN_BASE['name'], obj) and ends_ and max(ends_) > NUDGE_FIRST_END_MM and (res.get('axial') or {}).get('extent_mm'):
        # an end this high is a wedge depth cannot mistake (runs 31/32 after the insert: +7.7 / +8.1 mm; seated bars read
        # +1.5..+2.5 with depth's tilt): the map and a press before the nudge only confirmed it (~60 s, every press
        # blocked) - nudge first, the taps judge after it
        say(f"  [taps] {obj}: depth shows an end {max(ends_):+.1f} mm high (> {NUDGE_FIRST_END_MM:g}: a wedge) -> "
            f"one lateral nudge first, then the taps judge")
        ok, tp, blocked = False, None, True
    else:
        ok, tp = tap_recover(obj, ob, RUN_BASE['name'], say)
        if tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed())):
            return None                                      # the taps could not judge it: the depth ladder
        blocked = not ok and any(p.blocked for p in tp.points)
        if blocked:
            say(f"  [taps] {obj}: vertical presses blocked -> one lateral nudge, then the taps judge")
    if blocked:
        # the vertical presses are blocked (an end riding a lip springs back: 22:30 line_green +4.2 -> +2.7 -> +3.8 -> +3.8);
        # ONE lateral nudge frees it (22:35: depth ends +6.1 -> +0.6, taps then -0.8..-1.1), the taps judge after it -
        # never a second nudge on a depth reading (attempt 27: three, chasing +1.5-1.9 mm, moved the base ~50 mm)
        prev = os.environ.get('NUDGE_FREE_BASE')
        os.environ['NUDGE_FREE_BASE'] = '1'
        global NUDGE_TRIES
        tries0, NUDGE_TRIES = NUDGE_TRIES, 1
        try:
            nudge_into_slot(obj, args.observed_base, res, say, out_dir)
        finally:
            NUDGE_TRIES = tries0
            if prev is None:
                os.environ.pop('NUDGE_FREE_BASE', None)
            else:
                os.environ['NUDGE_FREE_BASE'] = prev
        ob2 = observe_base_pose(RUN_BASE['name']) or ob       # the push moves a free base: tap where it is now
        args.observed_base = args.measured_base = ob2
        ok, tp = tap_recover(obj, ob2, RUN_BASE['name'], say)
        if tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed())):
            # a point off the part right after a push: the base read lags the push (run 31: end+ on a U arm, +13.7 mm,
            # the other three points seated) - one more base read from its own view, one more map
            ob3 = observe_base_pose(RUN_BASE['name'])
            if ob3 is not None:
                args.observed_base = args.measured_base = ob3
                ok, tp = tap_recover(obj, ob3, RUN_BASE['name'], say)
        if tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed())):
            return None
    if ok and not blocked and (res.get('cross') or {}).get('centre_mm') is not None:
        # the across error the parts over it follow - from this depth read only if nothing moved the part since (after a
        # nudge it is the pre-nudge read: run 32 recorded [-0.2, -3.0] mm of a wedged bar the taps then found seated)
        _record_seat_check_error(obj, res)
    if ok:
        # with its own time and the points' places along the bar: ExecuteMate judges the ends from it while nothing has
        # moved since (seams.vision_ops.tap_ends)
        SEATED.mark(obj, True, how='taps', tap_map_mm={p.name: ('missed' if p.err_m is None else round(1000 * p.err_m, 2)) for p in tp.points},
                    tap_t=time.time(), tap_along_mm={p.name: round(1000 * p.along_m, 1) for p in tp.points})
    return ok


def _taps_mark(obj, args, say) -> bool:
    """Tap the part where drive-mate last measured the base; True (and the timed tap map in the ledger, which ExecuteMate
    judges the ends from) when the taps find it seated."""
    from seams.tap_probe import tap_recover
    ob_ = getattr(args, 'measured_base', None) or getattr(args, 'observed_base', None)
    if ob_ is None:
        return False
    say(f"  [taps] {obj}: the certifier read an end over its band - contact measures the ends")
    ok, tp = tap_recover(obj, ob_, RUN_BASE['name'], say)
    if not ok and (tp.off_part() or ({'mid', 'end-', 'end+'} & set(tp.missed()))):
        # taps beside the part: the base read they were placed from is stale by mm (run 50: end- and edge+ missed on a
        # bar depth read seated, ends +1.55 / +0.13 mm) - read the base where it is now and map once more
        ob2 = observe_base_pose(RUN_BASE['name'])
        if ob2 is not None:
            say(f"  [taps] {obj}: taps off/beside the part {tp.off_part() or tp.missed()} - base re-read, mapping again")
            args.observed_base = args.measured_base = ob2
            ok, tp = tap_recover(obj, ob2, RUN_BASE['name'], say)
    if ok:
        SEATED.mark(obj, True, how='taps', tap_map_mm={p.name: ('missed' if p.err_m is None else round(1000 * p.err_m, 2)) for p in tp.points},
                    tap_t=time.time(), tap_along_mm={p.name: round(1000 * p.along_m, 1) for p in tp.points})
    return bool(ok)


def seat_check_and_reseat(obj, state, args, say) -> bool:
    """The depth seat check (+ nudges / edge press); a pass is recorded in the verify ledger as a depth
    confirmation, which verify_seated accepts when the markers cannot see the part (2026-10-20: fork_orange,
    depth -0.08 mm from its CAD height, failed the run on 'marker not detected'). On a free base the taps judge
    the seat first (_seat_by_taps); the depth ladder runs only when they could not."""
    if (os.environ.get('SEAT_BY_TAPS', '1') == '1' and not BASE_FIXED_CFG()
            and getattr(args, 'observed_base', None) is not None):
        ok_t = _seat_by_taps(obj, args, say)
        if ok_t:
            return True
        if ok_t is None:
            say(f"  [seat-check] {obj}: the taps could not judge it -> the depth ladder")
        else:
            # run 33: after the nudge line_green sat +4.8 mm level and rolled (a long edge on the channel lip), its high
            # edge blocked at 4 N - the ladder's reseat with the insert grasp is the recovery left (run 31: it worked)
            say(f"  [seat-check] {obj}: the taps could not seat it -> the depth ladder (edge press, reseat)")
    ok = _seat_check_and_reseat(obj, state, args, say)
    if not ok:
        # the recovery ladder stops one read-sigma inside the certifier's band; when it has run out, a last reading inside
        # the band itself goes on to ExecuteMate, which decides (run 31: an end at +1.53 mm stopped the run, 0.03 mm over
        # the ladder's +-1.5)
        r_ = LAST_SEAT_RES.get(obj) or {}
        if r_.get('height_err_mm') is not None:
            from seams.acceptance import seat_inside
            inside, why = seat_inside(RUN_BASE['name'], obj, float(r_['height_err_mm']) / 1000.0, r_.get('end_heights_mm') or [])
            if inside:
                say(f"  [seat-check] {obj}: the ladder ran out inside the certifier's band ({why}) - ExecuteMate decides")
                ok = True
    if ok:
        SEATED.mark(obj, True, how='depth seat check')                  # merged: keeps rel_err_mm
    return ok


def _seat_check_and_reseat(obj, state, args, say) -> bool:
    """Depth-check the seated part; if it sits tilted or high, take it with the insert's own
    grasp (the pose the insert released it from), and insert it again. Fails closed."""
    check = _seat_check_accepted
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
        # a seat the check passes is not touched (operator 10-22: no nudges on a seated part; 14:02 the lip rule nudged an
        # ok line_green sideways)
        plan = cross_nudge_plan(res) if ('pitch_deg' in res and not res.get('ok')) else None
        if not BASE_FIXED_CFG() and os.environ.get('NUDGE_FREE_BASE', '0') != '1':
            plan = None                                       # no sideways push on a free base (see nudge_into_slot)
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
            _record_seat_check_error(obj, res)
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
                if ok:
                    _record_seat_check_error(obj, res2)
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

    def _clear_area_occupant(held):
        """Another part whose LATEST known position (tracker sighting or arm set-down, whichever is newer) lies within
        CLEAR_OCCUPIED_M of the clear area, or None. 10-22 05:29: u_brown was set down onto line_green lying there - the
        tracker had seen it there 9 minutes before."""
        from primitives.shared.config import CLEAR_AREA_XY
        latest = {}
        try:
            for k, v in json.load(open(TRACKER_JSON)).items():
                latest[k] = (float(v['t']), float(v['xyz'][0]), float(v['xyz'][1]))
        except Exception:
            pass
        try:
            for k, v in json.load(open(PLACED_JSON)).items():
                if k not in latest or float(v['t']) > latest[k][0]:
                    latest[k] = (float(v['t']), float(v['xy'][0]), float(v['xy'][1]))
        except Exception:
            pass
        for k, (t_, x_, y_) in latest.items():
            if k == held or k.startswith('base') or time.time() - t_ > 3600:
                continue
            if np.hypot(x_ - CLEAR_AREA_XY[0], y_ - CLEAR_AREA_XY[1]) < CLEAR_OCCUPIED_M:
                return k, (x_, y_), time.time() - t_
        return None

    def _gentle_place_down(step_idx, tag):
        """translate_object --place-down with the disassembly's landing (disassemble.place_at): the same
        move_to_clear_area, then move_down to LAND_SLOW_ABOVE_M above the CAD touch height of the held
        stance, then a slow move_cartesian that stops at LAND_FORCE_N. move_down --to-table alone aims at
        the closed-tip floor, below a part held by its top: yellow held by the bridge (recovery set-down,
        2026-10-19) hit the table at 37 N, 60 N peak. Returns None (caller runs the plain primitive) when
        the touch height cannot be computed."""
        occ = _clear_area_occupant(obj)
        if occ is not None:
            say(f"\n!!! clear area occupied: {occ[0]} last at ({1000*occ[1][0]:.0f}, {1000*occ[1][1]:.0f}) mm, {occ[2]:.0f} s ago "
                f"- {obj} not set down there (fail closed)")
            return {'result': 'failure', 'error': f'clear area occupied by {occ[0]}', 'movement_type': 'place_down'}
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
                snap_after = snapshot_robot_state()
                _record_snapshot(snap_fh, "after", f"{obj}#{step_idx}{tag} place_down", snap_after)
                xyz_ = (snap_after.get('tcp') or {}).get('xyz')
                if r.get('result') != 'failure' and xyz_:
                    # the part is where it was set down (2026-10-22: the regrasp's depth-pick view flew to u_orange's
                    # pick spot 0.8 m away - the stale tracker sighting - saw nothing, and the run stopped)
                    LAST_SEEN_XY[obj] = (float(xyz_[0]), float(xyz_[1])); PLACED_T[obj] = time.time()
                    record_placed(obj, xyz_)
                    say(f"  [{step_idx}] place_down: {obj} now at ({1000*xyz_[0]:.0f}, {1000*xyz_[1]:.0f}) mm (where it was set down)")
                return r
        if tool_name == 'move_to_grasp':
            age = camera_frame_age()
            if age is None or age > CAMERA_MAX_AGE_S:
                msg = (f"camera not streaming (last colour frame {'never' if age is None else f'{age:.1f} s ago'})"
                       " - check the D415 USB link (dmesg), restart realsense2_camera + localizer")
                say(f"\n!!! {msg}")
                return {'result': 'failure', 'error': msg}
        if (tool_name == 'translate_object' and kw.get('action') == 'insert' and one_node() and not args.dry_run
                and not BASE_FIXED_CFG()):
            # the servos fold one part's grip error (xy, image yaw) into the base pose they return - that is this insert's
            # AIM, not the base. Start every insert from the last MEASURED base (10-22 13:04: the in-hand re-read failed
            # as a partial view, line_green inherited u_orange's aim - its yaw 2.4 deg off - and wedged in the U openings)
            mb_ = args.measured_base if getattr(args, 'measured_base', None) is not None else getattr(args, 'observed_base', None)
            # AR3 (decision 10-22): drive-mate aligns the part (in-hand base read + servo) - the planner aims from the
            # measured base and sends its corrections; drive-mate returns the base it measured
            args.observed_base = mb_
        if _SAFETY_STOP:
            # 10-22 01:18: a protective stop inside hover_servo did not stop the run - drive-mate was called on a stopped arm
            # and waited 650 s. Nothing more is sent once the robot has left NORMAL safety.
            say(f"\n!!! robot {_SAFETY_STOP} - {tool_name} not started")
            return {'result': 'failure', 'error': f'robot {_SAFETY_STOP}'}
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
            if tgt is None:
                # no marker fallback (operator 10-23: "better to stop rather than do something harmful"): run 46's pick
                # fell back to the marker pose - 9-10 mm off at the low hover - and drove a finger onto u_brown at
                # -105 N (protective stop). No trusted grasp measurement -> no descent.
                say(f"\n!!! {obj}: no trusted grasp measurement (depth fit) - not picking")
                return {'result': 'failure', 'error': 'no trusted grasp measurement', 'label': f"{obj}#{step_idx}{tag} {tool_name}"}
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
        STATE_REF['args'] = args
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
            if not one_node():       # one node: the align funnel at the hover measures the held part where it matters
                held_yaw_calibration(obj, state, say, log_fh)  # one read where the arm is; no extra moves
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
        def _wedge_check(result):
            """None: not wedged. True/False: wedged; released and nudged -> seated or not. Called after EVERY failed
            attempt, before any lift: lifting a held part out of a wedge tripped a protective stop (cycle 3)."""
            if not (_is_bar(obj) and not _SAFETY_STOP and not args.dry_run):
                return None          # (10-22: the release-and-nudge is a recovery - it runs in one-node mode too)
            w = _wedged_near_seat(obj, t_step, result)
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

        if 'stopped at hover' in str(result.get('error', '')):
            say(f"  [{step_idx}] {obj}: stopped at the insert hover (STOP_AT_HOVER) - no descent")
            return False
        if result.get('result') == 'failure' and tool_name == 'translate_object' and kw.get('action') == 'insert':
            wc = _wedge_check(result)
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
                wc = _wedge_check(result)
                if wc is not None:
                    if wc:
                        print(f"\n  ✓ {obj} complete")
                    return wc

        if (result.get('result') == 'failure' and tool_name == 'translate_object'
                and kw.get('action') == 'insert' and insert_recoveries['n'] < INSERT_MAX_RECOVERIES
                and not _SAFETY_STOP and not one_node()):
            # (rule 2, one-node mode: not seated -> drive-mate's own retry -> stop; no regrasp ladder - 10-22 06:52 it
            # set line_green down and regrasped it after three failed attempts)
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
                    wc = _wedge_check(r_)                  # never lift a held part out of a wedge
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
            record_seat_offset(obj, t_step, result)
            seated_now = True

        # Tiny pause between primitives to let TF + topics settle (matches
        # run_assembly_step's settle-window discipline).
        time.sleep(0.5)

    if one_node() and not args.dry_run:
        # Operator 10-22: check EVERY seat right after its insert - drive-mate's at-rest TCP height cannot see a part
        # pitched in the jaws (08:15 line_green +2.2 mm at the TCP, pitched; the yellow then went onto it). The depth
        # seat check is the read that caught every tilted line_green on 10-14/15 (pitch 3.2-4.0 deg, one end +5.6..6.8
        # mm); its nudge / edge press / reseat is the recovery (not part of the compliant insert). Then ExecuteMate
        # certifies this mate from the seat view - mates are certified in order, one per seat.
        # ... only where a wedge is possible (CAD, cad_geometry.wedge_possible - operator 10-22: "reduce the unnecessary
        # things when not required"): line_green's end can rest on the base top looking seated; a U part or the yellow off
        # its pocket sits on the rim, far outside drive-mate's band. 11:47 a first-try seated u_orange (flat by touch:
        # ends +0.1 / -0.2 mm) read +2.4 mm at one end by depth, was tapped (base dragged 2.5 mm) and refused.
        from compliant_insertion_studio.wrapper.cad_geometry import wedge_possible
        wedgy = _seat_checked(obj) and getattr(args, 'seat_check', True) and wedge_possible(RUN_BASE['name'], obj)
        certify_ = os.environ.get('SEAMS_CERTIFY') == '1'
        # the certifier IS the per-seat check (its depth view judges the ends where a wedge is possible): the replay's own
        # depth seat check before it read the same seat a second time (~10 s a cycle, run 62) - it runs as the recovery
        # for a refusal, with its nudge. SEAT_CHECK_FIRST=1: the old order (seat check, then certify)
        if wedgy and (not certify_ or SEAT_CHECK_FIRST):
            if not seat_check_and_reseat(obj, state, args, say):
                return False
        if certify_:
            out = certify_via_execute_mate(obj, base_name or RUN_BASE['name'], getattr(args, 'observed_base', None),
                                           say, log_fh) or {}
            if wedgy and not SEAT_CHECK_FIRST and not out.get('placed') and out.get('refusal') == 'not_at_seat':
                say(f"  [execute_mate] {obj}: not_at_seat -> the seat check and its recovery (nudge / taps), then again")
                if not seat_check_and_reseat(obj, state, args, say):
                    return False
                out = certify_via_execute_mate(obj, base_name or RUN_BASE['name'], getattr(args, 'observed_base', None),
                                               say, log_fh) or {}
            if (not out.get('placed') and out.get('refusal') == 'not_at_seat' and wedge_possible(RUN_BASE['name'], obj)
                    and _taps_mark(obj, args, say)):
                # the seat check passed it by depth and skipped the taps; the certifier's own view read an end over the
                # band (run 40: line_green +2.68 mm, depth reads its ends high) - contact measures the ends, then ask again
                out = certify_via_execute_mate(obj, base_name or RUN_BASE['name'], getattr(args, 'observed_base', None),
                                               say, log_fh) or {}
            if not out.get('placed'):
                say(f"  [execute_mate] {obj}: {out.get('refusal') or 'no answer'} after the seat check - stopping")
                return False
    elif os.environ.get('SEAMS_CERTIFY') == '1' and not args.dry_run:
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
    if not args.dry_run:
        ensure_exposure(lambda m: print(m, flush=True))   # the room's light may have changed since the last run

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
            # ONE still window for every part from here (Speed T1: one-by-one observes waited 10.8 s for each part
            # home cannot see - 32-53 s per run, ASSEMBLY_STATE_MACHINE.md item 2)
            todo = [e.get('object_name') for e in entries
                    if not ((args.only and e.get('object_name') != args.only) or e.get('object_name') in skip)]
            seen_here = observe_many(todo, timeout_s=6.0, n_samples=6) if todo else {}
            if not getattr(args, 'skip_startup', False):
                NOT_FROM_HOME.update(n for n in todo if not seen_here.get(n))      # home cannot see these
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
                if not seen_here.get(obj):
                    unseen_home.append((entry, seq, bn)); continue        # not in this view: no 10 s wait
                entry['tool_sequence'], note = derive_sequence_for(obj, seq, bn, samples=seen_here[obj])
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
            # With --part-at-staging the clear area holds ONE part; the others wait in the staging region. Look from
            # the clear area first, then from the staging view for what it did not see (10-22 08:04: only the clear
            # area was viewed, u_orange/line_green/yellow each waited out a 15 s observe and ran the RECORDED
            # sequence - 48 s, and a part on another face would have run the wrong pass count).
            def _clear_view():
                r_ = run_subprocess(['python3', '-u', '-m', 'primitives.core.move_to_clear_area', '--mode', 'real'],
                                    label='view the clear area', timeout_s=90, log_fh=log_fh)
                time.sleep(1.5)
                return r_.get('result') != 'failure'
            views_ = ([('clear area', _clear_view)] if args.part_at_staging else []) + \
                     [('staging view', lambda: _staging_view(log_fh))]
            pending_ = list(unseen_home)
            for k_, (vname_, go_) in enumerate(views_):
                if not pending_ or not go_():
                    continue
                seen_there = observe_many([e['object_name'] for e, _, _ in pending_], timeout_s=8.0, n_samples=6)
                rest_ = []
                for entry, seq, bn in pending_:
                    s_ = seen_there.get(entry['object_name'])
                    if not s_ and k_ < len(views_) - 1:
                        rest_.append((entry, seq, bn)); continue          # try the next view, no 15 s wait here
                    if s_:
                        entry['tool_sequence'], note = derive_sequence_for(entry['object_name'], seq, bn, timeout_s=15.0,
                                                                           samples=s_)
                    else:
                        note = "not seen from any view -> recorded sequence (re-planned when the arm looks straight at it)"
                        entry['_planned_blind'] = (list(seq), bn)
                    msg = f"  [derive-sequence] {entry['object_name']} ({vname_}): {note}"
                    print(msg)
                    if log_fh: log_fh.write(msg + "\n")
                pending_ = rest_
            if unseen_home:
                if not getattr(args, 'base_from_camera', False):
                    # the base view comes next when the base is read from the camera: it is a straight high move
                    # from the staging view, home in between was a 4 s stop (ASSEMBLY_STATE_MACHINE.md item 7)
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
            from primitives.shared.config import ASSEMBLY_REGION_HALF as _ARH
            _lb = _last_base_pose(bname)          # over the base where it was last read (it moves: 9 cm off the
            _bx, _by = (_lb[0][0], _lb[0][1]) if _lb is not None else _ARC     # region centre left it half out of view)
            if abs(_bx - _ARC[0]) > _ARH[0] or abs(_by - _ARC[1]) > _ARH[1]:
                # the last read is outside this layout's assembly region: the base was moved (operator reset
                # 2026-10-21 16:15 - the view went to the old spot, the depth fit latched onto the parts lying there)
                _bx, _by = _ARC
            _tb = tracked(bname, TRACKER_TRUST_S)  # a recent still sighting from any view is where it is now
            if _tb is not None:
                _bx, _by = _tb[0], _tb[1]
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
            if ob is None and one_node():
                print(f"\n!!! {bname} not observed - drive-mate aims at a measured base only; stopping before any pick")
                return 1
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
                if getattr(args, 'observed_base', None) is not None and not one_node():
                    args.observed_base = reread_base(bname, args.observed_base, log_fh=log_fh)
                    # Arm is above the base: confirm every part seated so far really is.
                    seated_so_far = [e_.get('object_name') for e_ in entries if e_.get('object_name') in completed]
                    if seated_so_far and not verify_seated(seated_so_far, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")),
                                                                second_view=lambda: _home_view(log_fh)):
                        failed = seated_so_far[-1]
                        break
                if obj in NOT_FROM_HOME:
                    r = {'result': 'skipped'}       # home cannot see it: straight to the view above it (below)
                else:
                    r = run_subprocess(['python3', '-u', '-m', 'primitives.move_home', '--mode', 'real'],
                                       label=f'between-parts move_home ({obj})', timeout_s=60,
                                       log_fh=log_fh)
                if r.get('result') == 'failure':
                    failed = obj
                    break
            first_object = False
            if not args.dry_run and not _SAFETY_STOP and obj in FOLLOWS_SEATED and not one_node():
                record_follow_reads(obj, lambda m: (print(m, flush=True), log_fh and log_fh.write(m + "\n")),
                                    getattr(args, 'observed_base', None), log_fh, bool(args.part_at_staging))
            # --part-at-staging is a claim, not a fact (2026-10-20: line_brown was 38 cm away, in plain view from
            # home, and move_to_grasp gave up after 40 tries at the clear area): not seen there -> home, then above
            # its last position / the staging view.
            # A part the up-front observe saw away from the clear area is not looked for there nor from home: straight
            # above its last sighting (run 31: line_green paid clear-area 6 s + home 4 s + 10 s wait, then this view)
            from primitives.shared.config import CLEAR_AREA_XY as _CA
            _ls = LAST_SEEN_XY.get(obj)
            if (not args.dry_run and args.part_at_staging and not _SAFETY_STOP and _ls is not None
                    and np.hypot(_ls[0] - _CA[0], _ls[1] - _CA[1]) > PART_AWAY_FROM_CLEAR_M):
                msg = (f"  [view] {obj}: last seen at ({1000 * _ls[0]:.0f}, {1000 * _ls[1]:.0f}) mm, away from the clear "
                       f"area -> look above it")
                print(msg, flush=True)
                if log_fh: log_fh.write(msg + "\n")
                if not _part_view(obj, log_fh):
                    msg = f"\n!!! {obj}: not seen from above its last position nor from the staging view"
                    print(msg, flush=True)
                    if log_fh: log_fh.write(msg + "\n")
                    failed = obj
                    break
            elif (not args.dry_run and args.part_at_staging and not _SAFETY_STOP and not _seen_here(obj)):
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
                    and (obj in NOT_FROM_HOME or not _seen_here(obj))):
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
            # planned blind (no planning view saw it): plan it now, with the camera above it - 10-22 attempt 20 ran the
            # recorded 2 passes on an UPRIGHT yellow (1 pass), its extra set-down left it where the regrasp lost it
            if entry.get('_planned_blind') and not args.dry_run and not _SAFETY_STOP:
                seq0, bn0 = entry.pop('_planned_blind')
                # look straight down at it first (14:33 the check ran from over the base, before the view above the part)
                if not _seen_here(obj, 4.0):
                    _part_view(obj, log_fh)
                if _seen_here(obj, 8.0):
                    entry['tool_sequence'], note = derive_sequence_for(obj, seq0, bn0, timeout_s=10.0)
                    ts_, smp_ = SEEN_SAMPLES.get(obj, (0.0, None))
                    if note.startswith('observe failed') and smp_ and time.time() - ts_ < 30.0:
                        # the look that just saw it, from this same still pose: its samples plan it (run 60: the second
                        # look got none from a ~2 Hz localizer and the recorded 2 passes cost ~70 s on a 1-pass yellow)
                        entry['tool_sequence'], note = derive_sequence_for(obj, seq0, bn0, samples=smp_)
                        note += ' (from the samples of the look that saw it)'
                    msg = f"  [derive-sequence] {obj} (re-planned from the view above it): {note}"
                else:
                    msg = f"  [derive-sequence] {obj}: still not seen above it - recorded sequence"
                print(msg, flush=True)
                if log_fh: log_fh.write(msg + "\n")
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
        if one_node():
            pass        # one node: every seat was checked (depth) and certified (ExecuteMate) right after its insert
        elif (not failed and completed and not args.dry_run and not _SAFETY_STOP
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
    try:
        _sn = 'state_' + 'tap.py'                        # one standing subscriber for the per-step snapshots
        if subprocess.run(['pgrep', '-f', _sn], capture_output=True).returncode != 0:
            _TIMING['tap'] = subprocess.Popen(['python3', '-u', str(REPO_ROOT / 'ablations' / 'state_tap.py'),
                                               '--out', str(STATE_TAP_JSON)], env=make_env(),
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    print(f"  [timing] {timing_path}", flush=True)
    atexit.register(_stop_timing)


def _stop_timing():
    for k in ('rec', 'tracker', 'tap'):
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
# reads that ran untimed (10-25 s per part showed only as gaps, ASSEMBLY_STATE_MACHINE.md): base ICP/outline choice,
# the part depth fit at the pick, the line_green notch read, the one-view certification, the part view
choose_base_read = _timed(choose_base_read, 'base_read', lambda a, k: a[0])
depth_pick_target = _timed(depth_pick_target, 'depth_pick', lambda a, k: a[0])
record_follow_reads = _timed(record_follow_reads, 'follow_reads', lambda a, k: a[0])
certify_once_at_end = _timed(certify_once_at_end, 'certify_once', lambda a, k: ','.join(a[0]) if a else '')
_part_view = _timed(_part_view, 'part_view', lambda a, k: a[0])


if __name__ == '__main__':
    sys.exit(main())
