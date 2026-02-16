"""Signal phase completion to the MCP client.

Related files (phase signal pipeline):
  - triggers/signal_phase_complete.py (this file) — handles phase signals, invokes human for phase 3
  - elicitations/verify_assembly.py   — Pydantic schemas for human verification of real-world assembly

The agent calls this at the end of each assembly phase. The MCP client
watches for this tool's result to trigger the next phase.

Phase behaviors:
  - Phase 1 (Disassembly sequence discovery): Simple structured result.
  - Phase 2 (Assembly sequence discovery): On success without action, prompts
    the agent to choose "randomize" (reset scene and verify) or "reverified" (proceed
    to phase 3). On action="randomize", signals client to reset the scene.
    On action="reverified", signals client to proceed.
  - Phase 3 (Real-world execution): Internally invokes the verify_assembly
    elicitation so a human can verify the real-world assembly result.
"""

import os
import importlib.util
from typing import Literal, Optional, Dict, Any
from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession


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
        return _handle_phase_1(status, comment)
    elif phase == 2:
        return _handle_phase_2(status, action, comment)
    elif phase == 3:
        return await _handle_phase_3(status, comment, ctx, elicit_user_fn)


def _handle_phase_1(
    status: Literal["success", "failure"],
    comment: str,
) -> Dict[str, Any]:
    """Phase 1: Disassembly sequence discovery - simple structured result."""
    return {
        "type": "phase_complete",
        "phase": 1,
        "phase_name": "disassembly_sequence_discovery",
        "status": status,
        "comment": comment,
    }


def _handle_phase_2(
    status: Literal["success", "failure"],
    action: Optional[Literal["randomize", "reverified"]],
    comment: str,
) -> Dict[str, Any]:
    """Phase 2: Assembly sequence discovery.

    First call (no action): returns options for agent to choose.
    action="randomize": signals client to reset scene for verification.
    action="reverified": signals client to proceed to phase 3.
    """
    # Agent responding with an action
    if action is not None:
        return {
            "type": "phase_action",
            "phase": 2,
            "phase_name": "assembly_sequence_discovery",
            "action": action,
            "status": status,
            "comment": comment,
        }

    # First call - prompt agent to choose
    if status == "success":
        return {
            "type": "phase_complete",
            "phase": 2,
            "phase_name": "assembly_sequence_discovery",
            "status": status,
            "comment": comment,
            "requires_response": True,
            "message": "Assembly sequence recorded. Call this tool again with action='randomize' to reset the scene and verify the sequence, or action='reverified' to confirm verification is done and proceed to phase 3.",
            "options": ["randomize", "reverified"],
        }

    # Failure - no action needed
    return {
        "type": "phase_complete",
        "phase": 2,
        "phase_name": "assembly_sequence_discovery",
        "status": status,
        "comment": comment,
    }


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
    phase_result = {
        "type": "phase_complete",
        "phase": 3,
        "phase_name": "real_world_execution",
        "status": status,
        "comment": comment,
    }

    if elicit_user_fn is None:
        phase_result["human_verification"] = {
            "status": "skipped",
            "reason": "elicit_user function not available",
        }
        return phase_result

    try:
        # Dynamically load verify_assembly elicitation module
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        verify_assembly_path = os.path.join(script_dir, "elicitations", "verify_assembly.py")

        spec = importlib.util.spec_from_file_location(
            "elicitations.verify_assembly", verify_assembly_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Build message from phase result
        message = module.build_elicitation_message(phase_result)

        # Call elicit_user (same pattern as verify_clearance in server.py)
        elicit_response = await elicit_user_fn(ctx, "verify_assembly", message, phase_result)

        phase_result["human_verification"] = elicit_response

    except Exception as e:
        phase_result["human_verification"] = {
            "status": "error",
            "message": f"Human verification failed: {str(e)}",
        }

    return phase_result
