"""Run I/O shared by the planner (replay_real_assembly) and drive-mate's align step (seams/drive_align): primitives as
subprocesses with the protective-stop / program-drop watch, the robot-state snapshot, the run's timing file handle.
Moved out of the replay for AR3 (10-23) so drive-mate's process no longer imports the planner. The run-state flags
(_SAFETY_STOP, _ACTIVE_PROC, _INTERRUPTED) live HERE: other modules read them as cell_io.<name>."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

# Module-global handle to the currently running subprocess so the SIGINT
# handler can kill it (its whole process group) when the operator hits Ctrl-C.
_ACTIVE_PROC: Optional[subprocess.Popen] = None


_INTERRUPTED = False


_SAFETY_STOP: Optional[str] = None     # set when the robot leaves NORMAL safety: the run halts, no cleanup


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


_RESULT_RE = re.compile(r'__RESULT_JSON__\s*(.*?)\s*__END_RESULT_JSON__', re.DOTALL)


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
# Timing (2026-10-21, Speed track T0): one JSONL row per subprocess / read / seam call / part, plus an independent
# motion recorder (TCP speed). ablations/cycle_time.py splits the wall time into arm-moving and arm-still.
# Logging only: nothing here changes what the replay does.
# ---------------------------------------------------------------------------
_TIMING = {'fh': None, 'rec': None}


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
