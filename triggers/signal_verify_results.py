"""Verify scene state after orchestrator replay of discovered sequences.

Related files:
  - triggers/signal_phase_complete.py — gates on logged results (pre-replay)
  - triggers/signal_verify_results.py (this file) — gates on scene state (post-replay)

Called by the orchestrator after replaying the agent's logged tool_sequences.
The orchestrator passes its replay outcome as JSON; this handler runs
verify_assembly or verify_disassembly on the actual scene state, then
constructs an agent-facing message combining both.

Three response cases:
  1. Replay failed (tool call errored mid-execution)
     → result="failure", message tells agent which object/step to fix
  2. Replay succeeded + verification passed
     → result="success", phase can complete
  3. Replay succeeded but verification failed (scene state wrong)
     → result="failure", message tells agent sequences ran but result is wrong
"""

import json
from typing import Literal, Dict, Any, Optional, Callable


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

    # Case 1: Replay failed — tool call errored mid-execution
    if replay_status == "failure":
        failed_object = data.get("failed_object", "unknown")
        failed_step = data.get("failed_step", "unknown")
        error = data.get("error", "unknown error")
        completed = data.get("completed_objects", [])

        return {
            "result": "failure",
            "message": (
                f"Orchestrator replay failed on object '{failed_object}' "
                f"at step: {failed_step}. "
                f"Error: {error}. "
                f"Completed before failure: {completed}. "
                f"Revise the tool_sequence for '{failed_object}' "
                f"and signal phase {phase} complete again."
            ),
            "verification": verification,
        }

    # Case 2: Replay succeeded — check verification
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
