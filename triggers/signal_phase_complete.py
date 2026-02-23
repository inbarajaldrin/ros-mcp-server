"""Signal phase completion to the MCP client.

Related files (phase signal pipeline):
  - triggers/signal_phase_complete.py (this file) — handles phase signals, gates on logged results
  - elicitations/verify_assembly.py   — Pydantic schemas for human verification of real-world assembly (phase 4)

The agent calls this at the end of each assembly phase. The MCP client
watches for this tool's result to trigger the next phase.

Phase behaviors:
  - Phase 1 (Disassembly sequence discovery): Gates on Disassembly_*_results.json.
  - Phase 2 (Assembly sequence discovery): Gates on Assembly_*_results.json.
  - Phase 3 (Sim reverification): Gates on Assembly_*_results.json. Optionally
    accepts action="randomize" to reset scene for reverification, or
    action="reverified" to confirm optimized sequences work.
  - Phase 4 (Real-world execution): Internally invokes the verify_assembly
    elicitation so a human can verify the real-world assembly result.
"""

import os
from pathlib import Path
from typing import Literal, Optional, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession


def _get_logs_dir() -> Path:
    """Return the logs directory path, respecting MCP_CLIENT_OUTPUT_DIR."""
    base_output_dir = os.getenv("MCP_CLIENT_OUTPUT_DIR", "").strip()
    if base_output_dir:
        return Path(os.path.join(base_output_dir, "logs"))
    return Path(__file__).parent.parent / "logs"


def _has_assembly_results() -> bool:
    """Check if Assembly_*_results.json files exist in logs dir."""
    logs_dir = _get_logs_dir()
    return logs_dir.exists() and len(list(logs_dir.glob("Assembly_*_results.json"))) > 0


async def handle_phase_signal(
    phase: Literal[1, 2, 3, 4],
    status: Literal["success", "failure"],
    action: Optional[Literal["randomize", "reverified"]],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn=None,
) -> Dict[str, Any]:
    """Handle phase completion signal.

    Args:
        phase: Which phase completed (1, 2, 3, or 4).
        status: Whether the phase succeeded or failed.
        action: Phase 3 only (optional) - "randomize" or "reverified".
        comment: Optional comment (should explain failure).
        ctx: MCP Context (needed for phase 4 elicitation).
        elicit_user_fn: Reference to the elicit_user tool function from server.py.

    Returns:
        Structured result dict consumed by the MCP client.
    """
    # Enforce: action is only valid for phase 3
    if action is not None and phase != 3:
        return {
            "type": "error",
            "error": f"action parameter is only valid for phase 3, not phase {phase}.",
        }

    if phase == 1:
        result = _handle_phase_1(status, comment)
    elif phase == 2:
        result = _handle_phase_2(status, comment)
    elif phase == 3:
        result = _handle_phase_3(status, action, comment)
    elif phase == 4:
        result = await _handle_phase_4(status, comment, ctx, elicit_user_fn)

    # Stamp every return with phase/status so the MCP client can route it
    # Allow handlers to override status (e.g. gate check failing)
    effective_status = result.pop("status", status)
    envelope = {"phase": phase, "status": effective_status}
    if action is not None:
        envelope["action"] = action
    return {**envelope, **result}


def _handle_phase_1(
    status: Literal["success", "failure"],
    comment: str,
) -> Dict[str, Any]:
    """Phase 1: Disassembly sequence discovery.

    On success, gates completion on whether the agent has written disassembly
    results to the logbook.
    """
    if status != "success":
        return {}

    logs_dir = _get_logs_dir()
    has_results = (
        logs_dir.exists()
        and len(list(logs_dir.glob("Disassembly_*_results.json"))) > 0
    )

    if not has_results:
        return {
            "status": "failure",
            "requires_response": True,
            "message": "You have not logged any disassembly results. Use write_disassembly_results to log the grasp_id and gripper_state for each object you disassembled, then signal phase 1 complete again.",
        }

    return {"requires_response": False}


def _handle_phase_2(
    status: Literal["success", "failure"],
    comment: str,
) -> Dict[str, Any]:
    """Phase 2: Assembly sequence discovery.

    On success, gates completion on whether the agent has written assembly
    results to the logbook.
    """
    if status != "success":
        return {}

    if not _has_assembly_results():
        return {
            "status": "failure",
            "requires_response": True,
            "message": "You have not logged any assembly results. Use write_assembly_results to log the tool_sequence for each object you assembled, then signal phase 2 complete again.",
        }

    return {"requires_response": False}


def _handle_phase_3(
    status: Literal["success", "failure"],
    action: Optional[Literal["randomize", "reverified"]],
    comment: str,
) -> Dict[str, Any]:
    """Phase 3: Sim reverification.

    action="randomize": signals client to reset scene for reverification.
    action="reverified" (required to exit phase): gates on assembly results.
    No action: prompts agent to provide an action.
    """
    if status != "success":
        return {}

    # action="randomize" - signal client to reset scene
    if action == "randomize":
        return {"requires_response": True, "message": "Scene will be reset. Perform assembly with optimized sequences, then signal with action='reverified'."}

    # action="reverified" - gate on assembly results, exit phase
    if action == "reverified":
        if not _has_assembly_results():
            return {
                "status": "failure",
                "requires_response": True,
                "message": "No assembly results found. Ensure assembly results are logged before signaling reverified.",
            }
        return {"requires_response": False}

    # No action - prompt agent to use action
    return {
        "status": "failure",
        "requires_response": True,
        "message": "Phase 3 requires an action. Use action='reverified' to confirm sequences are verified, or action='randomize' to reset the scene first.",
    }


async def _handle_phase_4(
    status: Literal["success", "failure"],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn,
) -> Dict[str, Any]:
    """Phase 4: Real-world execution - invokes human verification elicitation.

    Loads the verify_assembly elicitation module to build the message and schema,
    then calls elicit_user to present the human with a verification form.
    """
    if elicit_user_fn is None:
        return {}

    try:
        import importlib.util

        # Dynamically load verify_assembly elicitation module
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        verify_assembly_path = os.path.join(script_dir, "elicitations", "verify_assembly.py")

        spec = importlib.util.spec_from_file_location(
            "elicitations.verify_assembly", verify_assembly_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Build message from phase result
        message = module.build_elicitation_message({"phase": 4, "status": status, "comment": comment})

        # Call elicit_user (same pattern as prepare_workspace in server.py)
        elicit_response = await elicit_user_fn(ctx, "verify_assembly", message, {"phase": 4, "status": status})

        return {"human_verification": elicit_response}

    except Exception as e:
        return {"human_verification": {"status": "error", "message": f"Human verification failed: {str(e)}"}}
