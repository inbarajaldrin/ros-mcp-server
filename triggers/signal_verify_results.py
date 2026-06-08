"""Verify scene state after orchestrator replay of discovered sequences.

Related files:
  - triggers/signal_phase_complete.py -- handles phase signals, gates on logged results
  - triggers/signal_verify_results.py (this file) -- post-replay verification, manages verify_rejection.json
  - resource.py -- write_assembly_results clears rejections via clear_rejection()
  - ablations/replay_verify.py -- orchestrator replay script, calls signal_verify_results

Called by the orchestrator after replaying the agent's logged tool_sequences.
The orchestrator passes its replay outcome as JSON; this handler runs
verify_assembly or verify_disassembly on the actual scene state, then
constructs an agent-facing message combining both.

Three response cases:
  1. Replay failed (tool call errored mid-execution)
     -> result="failure", message tells agent which object's LOG to fix
  2. Replay succeeded + verification passed
     -> result="success", phase can complete
  3. Replay succeeded but verification failed (scene state wrong)
     -> result="failure", message tells agent sequences ran but result is wrong
"""

import json
import os
from pathlib import Path
from typing import Literal, Dict, Any, Optional, Callable

MAX_REPLAY_ATTEMPTS_PER_OBJECT = 3


def _get_rejection_path() -> Path:
    """Return path to verify_rejection.json in the shared logs directory."""
    base = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
    if base:
        return Path(base) / "logs" / "verify_rejection.json"
    return Path(__file__).parent.parent / "logs" / "verify_rejection.json"


def _load_rejection() -> Dict:
    path = _get_rejection_path()
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"failed_objects": {}, "max_attempts_per_object": MAX_REPLAY_ATTEMPTS_PER_OBJECT}


def _save_rejection(data: Dict):
    path = _get_rejection_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def record_rejection(object_name: str, failed_step: str) -> int:
    """Record a replay failure for an object. Returns current attempt count."""
    data = _load_rejection()
    obj_state = data["failed_objects"].get(object_name, {"attempts": 0, "resolved": False})
    obj_state["attempts"] = obj_state.get("attempts", 0) + 1
    obj_state["resolved"] = False
    obj_state["last_failed_step"] = failed_step
    data["failed_objects"][object_name] = obj_state
    _save_rejection(data)
    return obj_state["attempts"]


def clear_rejection(object_name: str):
    """Clear rejection for an object after its results are rewritten."""
    data = _load_rejection()
    if object_name in data["failed_objects"]:
        data["failed_objects"][object_name]["resolved"] = True
        _save_rejection(data)


def mark_reprompted(object_name: str):
    """Mark that the agent has already been re-prompted to fix this object's log.
    Used by the first-failure @switch path: an object still unresolved AFTER a
    re-prompt means this model cannot self-correct its own log → switch model
    (keep context) rather than re-prompt the same model indefinitely. Survives
    record_rejection (which preserves the existing record) so the flag persists
    across the fix-and-resignal cycle even if `attempts` does not climb."""
    data = _load_rejection()
    rec = data["failed_objects"].setdefault(object_name, {"attempts": 0, "resolved": False})
    rec["reprompted"] = True
    _save_rejection(data)


def get_unresolved_rejections() -> Dict:
    """Return unresolved rejections {object_name: {attempts, last_failed_step}}."""
    data = _load_rejection()
    return {
        name: state for name, state in data["failed_objects"].items()
        if not state.get("resolved", False)
    }


def has_exhausted_object() -> Optional[str]:
    """Return the name of any object that has exhausted max replay attempts, or None."""
    data = _load_rejection()
    max_attempts = data.get("max_attempts_per_object", MAX_REPLAY_ATTEMPTS_PER_OBJECT)
    for name, state in data["failed_objects"].items():
        if not state.get("resolved", False) and state.get("attempts", 0) >= max_attempts:
            return name
    return None


def handle_verify_results(
    phase: Literal[1, 2],
    base_name: str,
    mode: str,
    replay_data: str,
    verify_disassembly_fn: Optional[Callable] = None,
    verify_assembly_fn: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Handle post-replay verification.

    Args:
        phase: 1 (disassembly) or 2 (assembly).
        base_name: Assembly base object name.
        mode: "sim" (replay verification is sim-only).
        replay_data: JSON string from orchestrator with replay outcome.
        verify_disassembly_fn: Callback(base_name, mode) -> verify result dict.
        verify_assembly_fn: Callback(base_name, mode) -> verify result dict.

    Returns:
        Dict with result, message, and verification details.
    """
    try:
        data = json.loads(replay_data)
    except (json.JSONDecodeError, TypeError) as e:
        return {
            "result": "failure",
            "message": f"Invalid replay_data JSON: {e}",
        }

    # Always check scene state regardless of replay outcome
    if phase == 1 and verify_disassembly_fn:
        verification = verify_disassembly_fn(base_name, mode)
    elif phase == 2 and verify_assembly_fn:
        verification = verify_assembly_fn(base_name, mode)
    else:
        verification = {}

    replay_status = data.get("replay", "failure")

    # Case 1: Replay failed -- tool call errored mid-execution
    if replay_status == "failure":
        failed_object = data.get("failed_object", "unknown")
        failed_step = data.get("failed_step", "unknown")
        completed = data.get("completed_objects", [])
        remaining = data.get("remaining_objects", [])
        completed_steps = data.get("completed_steps_for_object", [])
        total_steps = data.get("total_steps_for_object", [])

        # Record rejection and get attempt count
        attempts = record_rejection(failed_object, failed_step)
        max_attempts = MAX_REPLAY_ATTEMPTS_PER_OBJECT

        # Per-object ladder status: replay runs objects in order and stops at the
        # first failure, so report the whole ladder (passed / failed / not reached)
        # rather than spotlighting only the failed object.
        ladder_parts = [f"{o}: PASSED" for o in completed]
        ladder_parts.append(f"{failed_object}: FAILED at step {len(completed_steps)+1}")
        ladder_parts.extend(f"{o}: NOT REACHED" for o in remaining)
        ladder = " · ".join(ladder_parts)

        # Build step-by-step replay report for the failed object
        step_report = ""
        if completed_steps or total_steps:
            step_lines = []
            for i, s in enumerate(completed_steps):
                step_lines.append(f"  {i+1}. {s} -- OK")
            step_lines.append(f"  {len(completed_steps)+1}. {failed_step} -- FAILED")
            remaining_steps = total_steps[len(completed_steps)+1:]
            for i, s in enumerate(remaining_steps):
                step_lines.append(f"  {len(completed_steps)+2+i}. {s} -- not reached")
            step_report = (
                f"\n\nOrchestrator replay of your logged sequence for "
                f"'{failed_object}':\n" + "\n".join(step_lines)
            )

        return {
            "result": "failure",
            "message": (
                f"Your assembly was successful, but the tool_sequence you logged "
                f"for '{failed_object}' is incomplete. The orchestrator replayed "
                f"your logged sequence and failed at step {len(completed_steps)+1}. "
                f"You may be missing steps before or after this point. "
                f"Only log steps that actually moved the robot -- do not include "
                f"tool calls that failed during your execution. "
                f"Review your conversation history for the actual steps you "
                f"performed on every object not marked PASSED above and call "
                f"write_assembly_results with the complete sequence for each. "
                f"Do NOT re-assemble. "
                f"(Attempt {attempts}/{max_attempts}.)"
                f"\n\nReplay status (all objects, in order): {ladder}"
                f"{step_report}"
            ),
            "verification": verification,
        }

    # Case 2: Replay succeeded -- check verification
    if isinstance(verification, dict) and verification.get("result") == "success":
        return {
            "result": "success",
            "message": "All sequences verified. Replay completed and scene state confirmed.",
            "verification": verification,
        }

    # Case 3: Replay succeeded but scene state wrong
    if phase == 1:
        problem_objects = verification.get("still_assembled_objects", [])
        problem = f"Still assembled: {problem_objects}"
    else:
        problem_objects = verification.get("unassembled_objects", [])
        problem = f"Unassembled: {problem_objects}"

    return {
        "result": "failure",
        "message": (
            f"Orchestrator replay completed without errors but verification "
            f"failed. {problem}. Your logged sequences ran but did not "
            f"achieve the correct result. Revise your sequences and signal "
            f"phase {phase} complete again."
        ),
        "verification": verification,
    }
