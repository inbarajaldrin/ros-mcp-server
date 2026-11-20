"""drive-mate's latch file: ONE per-machine path, defined here and imported by the server, the replay and RealIO
(review w8-drivemate-gate r4 P2-1: a per-tree path failed open across worktrees - a server restarted from another
tree started unlatched and that tree's runs passed the startup check).

    ~/.local/state/fmb_cell/drive_mate_latch.json      ($FMB_STATE_DIR overrides the directory; tests use tmp)

PRESENCE is the latch: an empty, `{}` or unreadable file is latched ("latch file present"), so the server and every
startup check always agree.  Only a deliberate reset (the server's /ur5e/drive_mate/reset_latch, after its three
checks) removes it.  Writes are atomic (tmp + fsync + os.replace); a failed write is reported to the caller, which logs
it loudly; a state directory that can't be written is itself a reason to refuse a run (startup_refusal).
"""
import json
import os
import time
from pathlib import Path
from typing import Optional


def state_dir() -> Path:
    return Path(os.environ.get('FMB_STATE_DIR') or (Path.home() / '.local' / 'state' / 'fmb_cell'))


def latch_path() -> Path:
    return state_dir() / 'drive_mate_latch.json'


def read_latch(path=None) -> Optional[str]:
    """None when not latched (no file); else the reason (the file's, or 'latch file present')."""
    p = Path(path) if path is not None else latch_path()
    if not p.exists():
        return None
    try:
        reason = (json.load(open(p)) or {}).get('reason')
    except Exception:
        reason = None
    return reason or f'latch file present ({p})'


def write_latch(reason, path=None) -> Optional[str]:
    """Persist the latch; None, or why it could not be written."""
    p = Path(path) if path is not None else latch_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix('.tmp')
        with open(tmp, 'w') as fh:
            json.dump({'reason': reason, 'since': time.time()}, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        return None
    except Exception as e:
        return f'latch file {p} NOT written ({e!r})'


def remove_latch(path=None) -> Optional[str]:
    p = Path(path) if path is not None else latch_path()
    try:
        p.unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        return f'latch file {p} could not be removed ({e!r})'
    return None


def state_dir_refusal() -> Optional[str]:
    """None when the state directory is writable (a latch could be persisted); else why a run must not start."""
    d = state_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / f'.write_probe_{os.getpid()}'
        probe.write_text('x')
        probe.unlink()
        return None
    except Exception as e:
        return f'DRIVE_MATE_STATE_DIR_UNWRITABLE: {d} ({e!r}): a latch could not be persisted'
