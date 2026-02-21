"""Signal phase completion to the MCP client.

Related files (phase signal pipeline):
  - triggers/signal_phase_complete.py (this file) — handles phase signals, gates on logged results
  - elicitations/verify_assembly.py   — Pydantic schemas for human verification of real-world assembly (phase 3)

The agent calls this at the end of each assembly phase. The MCP client
watches for this tool's result to trigger the next phase.

Phase behaviors:
  - Phase 1 (Disassembly sequence discovery): Gates on Disassembly_*_results.json.
  - Phase 2 (Assembly sequence discovery): On success without action, prompts
    the agent to choose "randomize" (reset scene and verify) or "reverified" (proceed
    to phase 3). On action="randomize", signals client to reset the scene.
    On action="reverified", gates on Assembly_*_results.json before signaling
    client to proceed.
  - Phase 3 (Real-world execution): Internally invokes the verify_assembly
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


async def handle_phase_signal(
    phase: Literal[1, 2, 3],
    status: Literal["success", "failure"],
    action: Optional[Literal["randomize", "reverified"]],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn=None,
) -> Dict[str, Any]:
    """Handle phase completion signal.

    Args:
        phase: Which phase completed (1, 2, or 3).
        status: Whether the phase succeeded or failed.
        action: Phase 2 only - "randomize" to reset scene, "reverified" to proceed.
        comment: Optional comment (should explain failure).
        ctx: MCP Context (needed for phase 3 elicitation).
        elicit_user_fn: Reference to the elicit_user tool function from server.py.

    Returns:
        Structured result dict consumed by the MCP client.
    """
    # Enforce: action is only valid for phase 2
    if action is not None and phase != 2:
        return {
            "type": "error",
            "error": f"action parameter is only valid for phase 2, not phase {phase}.",
        }

    if phase == 1:
        result = _handle_phase_1(status, comment)
    elif phase == 2:
        result = _handle_phase_2(status, action, comment)
    elif phase == 3:
        result = await _handle_phase_3(status, comment, ctx, elicit_user_fn)

    # Stamp every return with phase/status so the MCP client can route it
    # Allow handlers to override status (e.g. phase 1 gate check failing)
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
    results to the logbook. Checks the shared output directory for
    Disassembly_*_results.json files (written by the Resources MCP server).
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
    action: Optional[Literal["randomize", "reverified"]],
    comment: str,
) -> Dict[str, Any]:
    """Phase 2: Assembly sequence discovery.

    First call (no action): returns options for agent to choose.
    action="randomize": signals client to reset scene for verification.
    action="reverified": gates on Assembly_*_results.json before proceeding.
    """
    # action="randomize" - just signal, no human needed
    if action == "randomize":
        return {}

    # action="reverified" - gate on assembly results before proceeding to phase 3
    if action == "reverified":
        logs_dir = _get_logs_dir()
        has_results = (
            logs_dir.exists()
            and len(list(logs_dir.glob("Assembly_*_results.json"))) > 0
        )

        if not has_results:
            return {
                "status": "failure",
                "requires_response": True,
                "message": "You have not logged any assembly results. Use write_assembly_results to log the tool_sequence for each object you assembled, then signal phase 2 complete with action='reverified' again.",
            }

        return {"requires_response": False}

    # First call - prompt agent to choose
    if status == "success":
        return {
            "requires_response": True,
            "message": "Assembly sequence recorded. Call this tool again with action='randomize' to reset the scene and verify the sequence, or action='reverified' to confirm verification is done and proceed to phase 3.",
            "options": ["randomize", "reverified"],
        }

    # Failure - no action needed
    return {}


async def _handle_phase_3(
    status: Literal["success", "failure"],
    comment: str,
    ctx: Context[ServerSession, None],
    elicit_user_fn,
) -> Dict[str, Any]:
    """Phase 3: Real-world execution - invokes human verification elicitation.

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
        message = module.build_elicitation_message({"phase": 3, "status": status, "comment": comment})

        # Call elicit_user (same pattern as prepare_workspace in server.py)
        elicit_response = await elicit_user_fn(ctx, "verify_assembly", message, {"phase": 3, "status": status})

        return {"human_verification": elicit_response}

    except Exception as e:
        return {"human_verification": {"status": "error", "message": f"Human verification failed: {str(e)}"}}
