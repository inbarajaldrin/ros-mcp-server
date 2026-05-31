#!/usr/bin/env python3
"""
instrument.py — observe-only telemetry sidecar for primitive timing, IK, and state.

Purpose (Track B baseline, 2026-05-30):
  Capture a measurement substrate from the deterministic no-LLM dry-run replays:
  per IK solve (PRIMARY), per-primitive timeline, and before/after kinematic state.
  Used to drive math-based IK optimization AFTER a baseline exists — this module
  only MEASURES, it never changes behavior.

Hard rules (enforced here so call sites stay trivial):
  - NEVER raises into the caller. Any internal error degrades to a stderr note + no-op.
  - Observe-only: emits monotonic timestamps + values it is handed. No new ROS
    subscriptions, no extra spin_once, no synchronous I/O inside hot callbacks —
    primitive-side records buffer in memory and flush ONCE at process exit.
  - Telemetry goes to a SIDECAR JSONL file (path from env), never to the primitive's
    stdout __RESULT_JSON__ block (the server returns that dict directly to the agent).
  - Kill switch: INSTRUMENT_OFF=1 makes everything a true no-op.

Two sides share one sidecar file (joined by call_id):
  * Server side (parent process): @timed_primitive decorator on _run_primitive +
    child_env() to push INSTRUMENT_CALL_ID / INSTRUMENT_SIDECAR into the subprocess.
    Sidecar path derives from MCP_CLIENT_OUTPUT_DIR.
  * Primitive side (subprocess): record(kind, **fields) buffers, flush() at atexit.
    Reads INSTRUMENT_SIDECAR / INSTRUMENT_CALL_ID from env.
"""

import os
import sys
import json
import time
import uuid
import atexit
import threading

_OFF = os.environ.get("INSTRUMENT_OFF", "").strip() == "1"

# ---- primitive-side (subprocess) env-driven sink -------------------------------
_SIDECAR = os.environ.get("INSTRUMENT_SIDECAR", "").strip()
_CALL_ID = os.environ.get("INSTRUMENT_CALL_ID", "").strip()

_buffer = []
_buf_lock = threading.Lock()
_flushed = False

# ---- server-side (parent) per-call context + seq -------------------------------
_local = threading.local()
_seq_lock = threading.Lock()
_seq = 0


def now():
    """Monotonic clock for durations (never wall-clock — see GPT review)."""
    return time.monotonic()


def _iso():
    # wall-clock stamp for human correlation only; durations use monotonic
    try:
        return time.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        return ""


def _safe(o):
    """JSON fallback for numpy / odd types. Never raises."""
    try:
        import numpy as _np
        if isinstance(o, _np.ndarray):
            return [float(x) for x in o.ravel().tolist()]
        if isinstance(o, _np.floating):
            return float(o)
        if isinstance(o, _np.integer):
            return int(o)
    except Exception:
        pass
    try:
        return str(o)
    except Exception:
        return None


def _append_to(path, records):
    """Append JSONL records to `path` atomically-ish (O_APPEND + flock). Never raises."""
    if not path or not records:
        return
    try:
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        blob = "".join(json.dumps(r, default=_safe) + "\n" for r in records)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            except Exception:
                pass
            os.write(fd, blob.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as e:
        try:
            sys.stderr.write("[instrument] sidecar append failed: %s\n" % e)
        except Exception:
            pass


def _next_seq():
    global _seq
    with _seq_lock:
        _seq += 1
        return _seq


# ================================================================================
# PRIMITIVE SIDE (subprocess): buffer in memory, flush once at exit
# ================================================================================

def primitive_enabled():
    return (not _OFF) and bool(_SIDECAR)


def record(kind, **fields):
    """Buffer one telemetry record (primitive subprocess side). Never raises.

    `kind` examples: "ik", "primitive", "snapshot", "event".
    """
    if not primitive_enabled():
        return
    try:
        rec = {
            "kind": kind,
            "call_id": _CALL_ID or None,
            "pid": os.getpid(),
            "t_mono": now(),
        }
        rec.update(fields)
        with _buf_lock:
            _buffer.append(rec)
    except Exception:
        pass


def flush():
    """Flush the in-memory buffer once. Registered at atexit; idempotent. Never raises."""
    global _flushed
    if _flushed or not primitive_enabled():
        return
    try:
        with _buf_lock:
            recs = list(_buffer)
            _buffer.clear()
            _flushed = True
        _append_to(_SIDECAR, recs)
    except Exception:
        pass


atexit.register(flush)


# ================================================================================
# SERVER SIDE (parent): decorator for _run_primitive + child env injection
# ================================================================================

def server_sidecar_path():
    """Per-run sidecar path under the host's output dir. Empty when not in a run."""
    base = os.environ.get("MCP_CLIENT_OUTPUT_DIR", "").strip()
    if not base:
        return ""
    return os.path.join(base, "timing", "primitive_timing.jsonl")


# Optional server-injected hook: snapshot(command_args) -> (state_dict|None, read_s).
# Lets the server read robot state (joints/gripper/EE) before+after each tool call
# WITHOUT the generic decorator needing ROS. Registered by the server at import.
_snapshot_hook = None


def register_snapshot_hook(fn):
    """Server registers a callable(command_args) -> (state|None, read_seconds)."""
    global _snapshot_hook
    _snapshot_hook = fn


def _take_snapshot(command_args):
    """Call the server's snapshot hook safely. Returns (state, read_s) or (None, None)."""
    if _OFF or _snapshot_hook is None:
        return None, None
    try:
        t0 = now()
        st = _snapshot_hook(command_args)
        return st, round(now() - t0, 4)
    except Exception:
        return None, None


def child_env():
    """Env overrides to push into the primitive subprocess (call_id + sidecar).

    Reads the call_id/sidecar established by @timed_primitive for THIS call.
    Returns {} when disabled or outside a measured call. Never raises.
    """
    try:
        if _OFF:
            return {}
        cid = getattr(_local, "call_id", None)
        sc = getattr(_local, "sidecar", None)
        if not cid or not sc:
            return {}
        return {"INSTRUMENT_CALL_ID": cid, "INSTRUMENT_SIDECAR": sc}
    except Exception:
        return {}


def timed_primitive(fn):
    """Decorator for the server's _run_primitive.

    Generates a per-call call_id + sidecar path, exposes them via child_env() for
    the subprocess env, and emits ONE coarse "call" record with subprocess_wall_s
    on every return path (try/finally). No-op when disabled. Never raises.
    """
    import functools

    @functools.wraps(fn)
    def wrapper(script_name, command_args="", *args, **kwargs):
        sidecar = "" if _OFF else server_sidecar_path()
        if not sidecar:
            return fn(script_name, command_args, *args, **kwargs)
        cid = uuid.uuid4().hex[:16]
        _local.call_id = cid
        _local.sidecar = sidecar
        seq = _next_seq()
        # Before-snapshot OUTSIDE the timing window (so it never inflates the
        # per-primitive wall measurement). Its own read_s is recorded.
        state_before, read_before_s = _take_snapshot(command_args)
        t0 = now()
        try:
            return fn(script_name, command_args, *args, **kwargs)
        finally:
            wall = round(now() - t0, 4)
            # After-snapshot also outside the timing window.
            state_after, read_after_s = _take_snapshot(command_args)
            try:
                _append_to(sidecar, [{
                    "kind": "call",
                    "call_id": cid,
                    "seq": seq,
                    "tool": script_name,
                    "args": command_args,
                    "subprocess_wall_s": wall,
                    "state_before": state_before if state_before is not None else "MISSING",
                    "state_after": state_after if state_after is not None else "MISSING",
                    "snapshot_read_before_s": read_before_s,
                    "snapshot_read_after_s": read_after_s,
                    "ts_iso": _iso(),
                }])
            except Exception:
                pass
            _local.call_id = None
            _local.sidecar = None

    return wrapper
