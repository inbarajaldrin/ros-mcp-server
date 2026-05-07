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
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _sigint_handler(signum, frame):
    """On Ctrl-C, kill the active subprocess group and abort.

    Without this the replay script's own KeyboardInterrupt won't propagate to
    the child primitive's process tree, leaving the robot in motion while
    the orchestrator crashes — a recipe for protective-stops.
    """
    global _INTERRUPTED
    _INTERRUPTED = True
    print("\n\n!!! SIGINT received — killing active subprocess and aborting replay !!!", flush=True)
    if _ACTIVE_PROC is not None and _ACTIVE_PROC.poll() is None:
        try:
            os.killpg(os.getpgid(_ACTIVE_PROC.pid), signal.SIGINT)
        except Exception:
            try:
                _ACTIVE_PROC.terminate()
            except Exception:
                pass
        # Give the child up to 3s to clean up, then kill hard
        try:
            _ACTIVE_PROC.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_ACTIVE_PROC.pid), signal.SIGTERM)
            except Exception:
                try:
                    _ACTIVE_PROC.kill()
                except Exception:
                    pass
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


def make_env() -> dict:
    env = os.environ.copy()
    env['PYTHONPATH'] = (
        f"{REPO_ROOT}:{env['PYTHONPATH']}" if 'PYTHONPATH' in env else str(REPO_ROOT)
    )
    return env


def run_subprocess(argv: list[str], label: str, timeout_s: int = 600,
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
    try:
        deadline = time.time() + timeout_s
        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)
            captured.append(line)
            if log_fh is not None:
                log_fh.write(line)
                log_fh.flush()
            if _INTERRUPTED:
                break
            if time.time() > deadline:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.kill()
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
        _ACTIVE_PROC = None
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

    def update_from_result(self, tool_name: str, result: dict) -> None:
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

def build_argv(tool_name: str, kw: dict, state: AssemblyState,
               base_offset_xy: Optional[Tuple[float, float]] = None) -> Tuple[list[str], int]:
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
                     '--mode', 'real'], 60

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


def replay_object(entry: dict, state: AssemblyState, args: argparse.Namespace,
                   log_fh=None, snap_fh=None) -> bool:
    obj = entry['object_name']
    sequence = entry.get('tool_sequence', []) or []
    print(f"\n{'='*72}\n  Replaying object: {obj}  ({len(sequence)} steps)\n{'='*72}")

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

        try:
            argv, timeout_s = build_argv(tool_name, kw, state,
                                          base_offset_xy=args.base_offset_xy)
        except Exception as e:
            print(f"  [{step_idx}] BUILD FAILED ({e}); aborting.")
            return False

        if args.dry_run:
            print(f"  [{step_idx}] [DRY] {' '.join(argv)}")
            continue

        label = f"{obj}#{step_idx} {tool_name}"

        # Snapshot joint+TCP state BEFORE the primitive (debug-quality record)
        snap_before = snapshot_robot_state()
        _record_snapshot(snap_fh, "before", label, snap_before)

        result = run_subprocess(argv, label=label, timeout_s=timeout_s, log_fh=log_fh)
        state.update_from_result(tool_name, result)

        # Snapshot joint+TCP state AFTER the primitive
        snap_after = snapshot_robot_state()
        _record_snapshot(snap_fh, "after", label, snap_after)

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

        if result.get('result') == 'failure':
            print(f"\n!!! STEP FAILED: {step}")
            print(json.dumps(result, indent=2))
            return False

        # Tiny pause between primitives to let TF + topics settle (matches
        # run_assembly_step's settle-window discipline).
        time.sleep(0.5)

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
    p.add_argument('--base-offset-xy', type=float, nargs=2, metavar=('DX', 'DY'),
                   default=None,
                   help="xy offset (m) injected into the insert step's base pose. Useful for "
                        "tuning per-object insert location empirically.")
    p.add_argument('--dry-run', action='store_true',
                   help="Print the planned subprocess argv without executing.")
    p.add_argument('--initial-held-quat', type=float, nargs=4,
                   metavar=('QX', 'QY', 'QZ', 'QW'), default=None,
                   help="Optional initial held_quat to seed the chain (e.g. when starting "
                        "mid-sequence with a part already in the gripper).")
    p.add_argument('--skip-startup', action='store_true',
                   help="Skip the move_home + gripper-open startup phase. Use when the "
                        "robot is already in a known clean state.")
    p.add_argument('--log-dir', type=str, default=str(LOG_ROOT),
                   help="Directory for the per-run log file. Default: ablations/logs/.")
    args = p.parse_args()

    # Install SIGINT handler so Ctrl-C kills the active subprocess tree
    # immediately instead of letting it run to completion (which previously
    # caused protective stops because the operator's interrupt didn't
    # propagate to the child primitive's motion).
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    if args.base_offset_xy is not None:
        args.base_offset_xy = tuple(args.base_offset_xy)

    json_path = Path(args.assembly_json)
    if not json_path.exists():
        print(f"ERROR: assembly JSON not found: {json_path}")
        return 2
    d = json.load(open(json_path))
    entries = sorted(d.get('assembly_order', []), key=lambda e: e.get('assembly_order', 0))
    if not entries:
        print("ERROR: no assembly_order entries in JSON")
        return 2

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

        completed: list[str] = []
        failed: Optional[str] = None
        for entry in entries:
            obj = entry.get('object_name')
            if args.only and obj != args.only:
                continue
            ok = replay_object(entry, state, args, log_fh=log_fh, snap_fh=snap_fh)
            if ok:
                completed.append(obj)
            else:
                failed = obj
                break

        # Cleanup: move_home (always, success or failure path). Skipped on
        # --dry-run AND when --stop-before / --stop-after fired (pause means
        # the robot stays at the halted pose for operator inspection or
        # follow-on motion).
        if not args.dry_run and not _STOP_BEFORE_FIRED and not _STOP_AFTER_FIRED:
            header = f"\n{'='*72}\n  CLEANUP: move_home\n{'='*72}\n"
            print(header, end='')
            if log_fh: log_fh.write(header)
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


if __name__ == '__main__':
    sys.exit(main())
